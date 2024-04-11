#include <iostream>
#include <random>

#include <Eigen/Dense>

#include "lbfgs.hpp"

int cnt=0;
int formation_type = 0;
std::vector<Eigen::Vector2d> drones;

Eigen::MatrixXd L_hat;

void generateDrones()
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

class test
{
public:
    double similarityCal(const Eigen::MatrixXd& L, const Eigen::MatrixXd& L_hat)
    {
        return pow((L-L_hat).norm(), 2);
    }

    void graphCal(const std::vector<Eigen::Vector2d>& drones, Eigen::MatrixXd& L, Eigen::MatrixXd& A, Eigen::VectorXd& D)
    {
        A = Eigen::MatrixXd::Zero(drones.size(), drones.size());
        D = Eigen::VectorXd::Zero(drones.size());
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

    static double costFunctionCallback(void *func_data, const double* x, double* grad, const int n)
    {
        test *opt = reinterpret_cast<test *>(func_data);

        // (grad)[0] = 4*pow(x[0], 3) - 3*pow(x[0], 2) - 14*x[0] + 1;
        // (grad)[1] = 20 * x[1];
        // grad[0] = x[0] * (1.73928*exp(-0.826446*pow(x[0], 2))*pow(x[1], 2)- 0.915091*exp(-1.65289*pow(x[0], 2))+0.01)-0.05;
        // grad[1] = 4*pow(x[1], 3) - 2.10453*exp(-0.826446*pow(x[0], 2))*x[1];
        // double a = 1/(pow(0.55, 2) * 2 * M_PI) * exp(-pow(x[0], 2)/1.21);
        // grad[0] = 4/1.21*x[0] * (pow(x[1], 2) - a)*a+0.01*(x[0]-5);
        // grad[1] = 4*x[1]*(pow(x[1], 2)-a);

        // double cost = pow(x[0], 2) + pow(x[1], 2);
        // double cost = pow(x[0], 4) - pow(x[0], 3) - 7*pow(x[0], 2) + x[0] + 10 * pow(x[1], 2) + 6;
        // double sigma = 1.1;
        // double g = (1/sqrt(2*M_PI)*sigma)*exp(-pow(x[0], 2)/ (2*pow(sigma, 2)));
        // double f = pow((x[1]+2*g), 2) * pow((x[1]-2*g), 2);
        // double cost = f + 0.005 * pow((x[0]-5), 2);
        
        auto new_drones = drones;
        new_drones[new_drones.size()-1] = Eigen::Vector2d(x[0], x[1]);
        Eigen::MatrixXd L;
        Eigen::MatrixXd A;
        Eigen::VectorXd D;
        opt->graphCal(new_drones, L, A, D);
        double cost = opt->similarityCal(L, L_hat);

        Eigen::MatrixXd DLhat = L - L_hat;
        double b0 = 0;
        for (int i=0; i<drones.size(); i++)
            b0+=A(drones.size()-1, i) * DLhat(drones.size()-1, i)/sqrt(D(i));
        
        b0*=2*pow(D(drones.size()-1), -1.5);

        int iter = 0;
        Eigen::VectorXd dfde = Eigen::VectorXd::Zero(drones.size() - 1);
        for( int i = 0; i < drones.size(); i++ ){
            if( i != drones.size()-1){
                for( int k = 0; k < drones.size(); k++ ){
                    dfde(iter) += A(i, k) * DLhat(i, k) / sqrt(D(k));
                }
                dfde(iter) = 2 * pow(D(i), -1.5) * dfde(iter) + b0 + 4 * ( -1/(sqrt(D(i))*sqrt(D(drones.size()-1)))) * DLhat(i, drones.size()-1);
                iter++;
            }
        }

        Eigen::MatrixXd dedp(drones.size() - 1, 2);

        iter = 0;
        for( int i = 0; i < drones.size(); i++ ){
            if( i != drones.size()-1 ){
                for( int k = 0; k < 2; k++ ){
                    dedp(iter, k) = new_drones[drones.size()-1](k)-new_drones[i](k);
                }
                iter++;
            }
        }

        auto grad_eigen = dfde.transpose() * dedp;
        grad[0] = grad_eigen(0);
        grad[1] = grad_eigen(1);

        // std::cout << "cost: " << cost << " x0: " << x[0] << " x1: " << x[1] << " grad0: " << grad[0] << " grad1: " << grad[1] << std::endl;

        return cost;
    }

    double lbfgs_optimize()
    {
        int variable_num = 2;
        double q[variable_num];
        double final_cost;

        std::random_device rd;
        std::mt19937 gen(rd());
    
        std::uniform_real_distribution<double> dis(-10, 10);

        q[0] = (dis(gen));
        // q[1] = (0.000001);
        q[1] = dis(gen);

        Eigen::MatrixXd A;
        Eigen::VectorXd D;
        graphCal(drones, L_hat, A, D);

        lbfgs::lbfgs_parameter_t lbfgs_params;
        lbfgs::lbfgs_load_default_parameters(&lbfgs_params);
        lbfgs_params.mem_size = 16;
        // lbfgs_params.g_epsilon = 0.01;
        lbfgs_params.g_epsilon = 1e-6;
        lbfgs_params.min_step = 1e-32;

        lbfgs_params.max_iterations = 20;

        int result = lbfgs::lbfgs_optimize(variable_num, q, &final_cost, test::costFunctionCallback, NULL, NULL, this, &lbfgs_params);
        printf("The optimization result is : %s, optimal variable: (%lf, %lf), final cost: %lf\n",
                     lbfgs::lbfgs_strerror(result), q[0], q[1], final_cost);
        if (final_cost < 1e-2 && (Eigen::Vector2d(q[0], q[1]) - drones[drones.size()-1]).norm() < 0.01)
            cnt++;
        return final_cost;
    }
};

int main(int argc, char** argv)
{
    test t;
    generateDrones();

    for (size_t i=0; i<1e5; ++i)
    t.lbfgs_optimize();

    std::cout << "cnt: " << cnt << std::endl;

    return 0;
}