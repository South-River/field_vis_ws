#include <vector>
#include <random>

#include <ros/ros.h>

#include <Eigen/Dense>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/PointCloud2.h>

ros::Publisher pub_obstacles_, pub_esdf_, pub_drones_, pub_formation_field_;

int formation_type = 0;

double obstacle_radius = 0.5;

double formation_field_limit = 10.0;

void graphCal(const std::vector<Eigen::Vector2d>& drones, Eigen::MatrixXd& L)
{
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(drones.size(), drones.size());
    Eigen::VectorXd D = Eigen::VectorXd::Zero(drones.size());
    L = Eigen::MatrixXd::Zero(drones.size(), drones.size());

    for (size_t i=0; i<drones.size(); i++)
        for (size_t j=0; j<drones.size(); j++)
        {
            A(i, j) = (drones[i] - drones[j]).cwiseAbs2().sum();
            D(i) += A(i, j);
        }

    for (size_t i=0; i<drones.size(); i++)
        for (size_t j=0; j<drones.size(); j++)
        {
            if (i==j)
                L(i, j) = 1;
            else
                L(i, j) = -A(i, j) * pow(D(i), -0.5) * pow(D(j), -0.5);
        }
}

double similarityCal(const Eigen::MatrixXd& L, const Eigen::MatrixXd& L_hat)
{
    return pow((L-L_hat).norm(), 2);
}


void visDrones(const std::vector<Eigen::Vector2d>& drones)
{
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time::now();
    marker.id = 0;
    marker.type = visualization_msgs::Marker::SPHERE_LIST;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;

    double radius = 0.2;
    marker.scale.x = radius;
    marker.scale.y = radius;
    marker.scale.z = radius;
    marker.color.a = 1.0;
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;

    geometry_msgs::Point pt;
    for (auto& drone: drones)
    {        
        pt.z = 0.5;
        
        pt.x = drone(0) - 0.1; pt.y = drone(1) - 0.1;
        marker.points.push_back(pt);

        pt.x = drone(0) + 0.1; pt.y = drone(1) - 0.1;
        marker.points.push_back(pt);

        pt.x = drone(0) + 0.1; pt.y = drone(1) + 0.1;
        marker.points.push_back(pt);

        pt.x = drone(0) - 0.1; pt.y = drone(1) + 0.1;
        marker.points.push_back(pt);
    }

    pub_drones_.publish(marker);
}

void visObstacles(const std::vector<Eigen::Vector2d>& obss)
{
    visualization_msgs::MarkerArray marker_array;

    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time::now();
    marker.ns = "obstacles";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::CYLINDER;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = obstacle_radius;
    marker.scale.y = obstacle_radius;
    marker.scale.z = 1.0;
    marker.color.a = 0.48;
    marker.color.r = 0.00;
    marker.color.g = 0.00;
    marker.color.b = 1.00;

    for (size_t i = 0; i < obss.size(); i++)
    {
        marker.id = i;
        marker.pose.position.x = obss[i](0);
        marker.pose.position.y = obss[i](1);
        marker.pose.position.z = 0.5;
        marker_array.markers.push_back(marker);
    }

    pub_obstacles_.publish(marker_array);
}

pcl::PointCloud<pcl::PointXYZI> calESDF(const std::vector<Eigen::Vector2d>& obss)
{
    pcl::PointCloud<pcl::PointXYZI> cloud;
    pcl::PointXYZI pt;

    const double min_dist = -obstacle_radius; 
    const double max_dist = 5.0;

    double step = 0.1;

    for (double x = -10.0; x <= 10.0; x += step)
    {
        for (double y = -10.0; y <= 10.0; y += step)
        {
            double dist = 100.0;
            for (const auto& obs:obss)
            {
                double d = (obs - Eigen::Vector2d(x, y)).norm() - obstacle_radius;
                if (d < dist)
                {
                    dist = d;
                }
            }
            if (dist < min_dist)
            {
                dist = min_dist;
            }
            if (dist > max_dist)
            {
                dist = max_dist;
            }
            pt.x = x;
            pt.y = y;
            pt.z = -dist;
            pt.intensity = (dist - min_dist) / (max_dist - min_dist);
            cloud.push_back(pt);
        }
    }

    cloud.width = cloud.points.size();
    cloud.height = 1;
    cloud.is_dense = true;
    cloud.header.frame_id = "map";

    return cloud;
}

void visESDF(const pcl::PointCloud<pcl::PointXYZI>& cloud)
{
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(cloud, cloud_msg);

    pub_esdf_.publish(cloud_msg);
}

pcl::PointCloud<pcl::PointXYZI> calFormationField(std::vector<Eigen::Vector2d> drones)
{
    Eigen::MatrixXd L_des;
    graphCal(drones, L_des);

    pcl::PointCloud<pcl::PointXYZI> cloud;
    pcl::PointXYZI pt;

    double min_dist = std::numeric_limits<double>::max(); 
    double max_dist = std::numeric_limits<double>::min();

    double step = 0.01;
    for (double x = -formation_field_limit; x <= formation_field_limit; x += step)
        for (double y = -formation_field_limit; y <= formation_field_limit; y += step)
        {
            Eigen::MatrixXd L;
            drones[drones.size()-1] = Eigen::Vector2d(x, y);
            graphCal(drones, L);
            double dist = similarityCal(L, L_des);
            
            pt.x = x; 
            pt.y = y;
            pt.z = dist;
            cloud.push_back(pt);
            min_dist = std::min(min_dist, dist);
            max_dist = std::max(max_dist, dist);
        }
    
    for (auto& pt: cloud.points)
    {
        pt.intensity = (pt.z - min_dist)/(max_dist-min_dist);        
        pt.z *= 10;
    }

    cloud.width = cloud.points.size();
    cloud.height = 1;
    cloud.is_dense = true;
    cloud.header.frame_id = "map";

    return cloud;
}

void visFormationField(const pcl::PointCloud<pcl::PointXYZI>& cloud)
{
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(cloud, cloud_msg);

    pub_formation_field_.publish(cloud_msg);
}

void generateRandomObstacles(std::vector<Eigen::Vector2d>& obss, const int& num_obstacles)
{
    std::random_device rd;
    std::mt19937 gen(rd());
  
    std::uniform_real_distribution<double> dis(-10, 10);

    for (auto cnt = 0; cnt < num_obstacles;)
    {
        Eigen::Vector2d new_obs(dis(gen), dis(gen));
        bool flag = false;
        for (auto& obs:obss)
        {
            auto dis = (obs - new_obs).norm();
            if (dis < 1.0)
            {
                flag = true;
                break;
            }
        }
        if (!flag)
        {
            cnt++;
            obss.push_back(new_obs);
        }
    }
}

void generateDrones(std::vector<Eigen::Vector2d>& drones)
{
    if (formation_type == 0)
    {
        /* Triangle */
        drones.push_back(Eigen::Vector2d(+2.0, +0.0));
        drones.push_back(Eigen::Vector2d(-2.0, +0.0));
        drones.push_back(Eigen::Vector2d(+0.0, +2.0));
    }
    else if (formation_type == 1)
    {
        /* Square */
        drones.push_back(Eigen::Vector2d(+1.0, 0.0));
        drones.push_back(Eigen::Vector2d(-1.0, 0.0));
        drones.push_back(Eigen::Vector2d(0.0, +1.0));
        drones.push_back(Eigen::Vector2d(0.0, -1.0));
    }
    else if (formation_type == 2)
    {
        /* Star */
        drones.push_back(Eigen::Vector2d(1.0, 0.0));
        drones.push_back(Eigen::Vector2d(0.309, 0.951));
        drones.push_back(Eigen::Vector2d(0.309,-0.951));
        drones.push_back(Eigen::Vector2d(-0.809, 0.588));
        drones.push_back(Eigen::Vector2d(-0.809, -0.588));
    }
    else if (formation_type == 3)
    {
        /* Hexagon */
        drones.push_back(Eigen::Vector2d(+0.0, 0.0));
        drones.push_back(Eigen::Vector2d(1.7321, -1.0));
        drones.push_back(Eigen::Vector2d(0.0, -2.0));
        drones.push_back(Eigen::Vector2d(-1.7321, -1.0));
        drones.push_back(Eigen::Vector2d(-1.7321, 1.0));
        drones.push_back(Eigen::Vector2d(0.0, 2.0));
        drones.push_back(Eigen::Vector2d(1.7321, 1.0));
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "field_vis");
    ros::NodeHandle nh_("~");

    nh_.param("formation_type", formation_type, 0);

    pub_obstacles_ = nh_.advertise<visualization_msgs::MarkerArray>("/obstacles", 1);
    pub_esdf_ = nh_.advertise<sensor_msgs::PointCloud2> ("/esdf", 1);
    pub_formation_field_ = nh_.advertise<sensor_msgs::PointCloud2> ("/formation_field", 1);
    pub_drones_ = nh_.advertise<visualization_msgs::Marker>("/drones", 1);

    ROS_INFO("Hello! formation type: %d", formation_type);

    std::vector<Eigen::Vector2d> obss;
    generateRandomObstacles(obss, 150);

    std::vector<Eigen::Vector2d> drones;
    generateDrones(drones);

    auto esdf = calESDF(obss);
    auto formation_field = calFormationField(drones);

    ROS_INFO("Start visualization...");
    ros::Rate r(0.5);

    while (ros::ok())
    {
        visObstacles(obss);
        visESDF(esdf);
        visDrones(drones);
        visFormationField(formation_field);
        r.sleep();
        ros::spinOnce();
    }

    ros::spin();
    return 0;
}