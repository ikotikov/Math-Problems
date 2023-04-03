//
//  main.cpp
//  Rung
//
//  Created by Ivan Kotikov on 21.04.2022.
//

#include <iostream>
#include <vector>
#include <map>
#include <math.h>
#include <fstream>
#include <chrono>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
using namespace std::chrono;


double f1 (double t, VectorXd y) {
    double f = 2 * t * y(0) * log(max(y(1), 0.001));
    return f;
}

double f2 (double t, VectorXd y) {
    double f = -2 * t * y(1) * log(max(y(0), 0.001));
    return f;
}

VectorXd y_ext (double t) {
    VectorXd yext {{exp(sin(t * t)), exp(cos(t * t))}};
    return yext;
}

double GLobal (vector<VectorXd> y, vector<double> t) {
    double max_diff = 0;
    for (int i = 0; i < y.size(); ++i) {
        double ti = t[i];
        double nor = (y[i] - y_ext(ti)).norm();
        if (nor > max_diff) {
            max_diff = nor;
        }
    }
    return max_diff;
}


double AutoStep (const double& h0, const double& tol, const double& err, const double fac, const double& facmax, const double& facmin) {
//    double facmin = 0.5;
//    double fac = 0.8;
    
    double h = h0 * min(facmax, max(facmin, fac * pow(tol/err, 1./5.)));
    
    return h;
}

vector<VectorXd> k_coeff (double t, VectorXd y, double h) {
    vector<vector<double>> a = {{}, {1./2.}, {0., 1./2.}, {0., 0., 1.}};
    vector<double> c = {0., 1./2., 1./2., 1.};
    vector<VectorXd> k(4);
    VectorXd kv1(2);
    kv1(0) = f1(t, y);
    kv1(1) = f2(t, y);
    k[0] = kv1;
    
    VectorXd kv2(2);
    kv2(0) = f1(t + c[1] * h, y + h * a[1][0] * k[0]);
    kv2(1) = f2(t + c[1] * h, y + h * a[1][0] * k[0]);
    k[1] = kv2;
    
    VectorXd kv3(2);
    kv3(0) = f1(t + c[2] * h, y + h * (a[2][0] * k[0] + a[2][1] * k[1]));
    kv3(1) = f2(t + c[2] * h, y + h * (a[2][0] * k[0] + a[2][1] * k[1]));
    k[2] = kv3;
    
    VectorXd kv4(2);
    kv4(0) = f1(t + c[3] * h, y + h * (a[3][0] * k[0] + a[3][1] * k[1] + a[3][2] * k[2]));
    kv4(1) = f2(t + c[3] * h, y + h * (a[3][0] * k[0] + a[3][1] * k[1] + a[3][2] * k[2]));
    k[3] = kv4;

    return k;
}

VectorXd RK_Step (vector<VectorXd> k, VectorXd y, double h) {
    vector<double> b = {1./6., 2./6., 2./6., 1./6.};
    VectorXd f = y + h * (b[0] * k[0] + b[1] * k[1] + b[2] * k[2] + b[3] * k[3]);
    
    return f;
}

void RungeKutta (vector<VectorXd>& y, vector<double>& t, double facmax, double facmin, double fac, double tol, double N_Steps, vector<double>& err_g) {
    double Nv = 5.0;
    
    double t1 = Nv * 0.1;
    double t2 = t1 + 4.0;
    t.push_back(t1);
    
    double y10 = exp(sin(t1 * t1));
    double y20 = exp(cos(t1 * t1));
    VectorXd y0 {{y10, y20}};
    y.push_back(y0);
    
    double t_curr = t1;
    double h_curr = 0.1;
    
    while (t_curr <= t2) {
        // Global
        // y[y.size() - 1] -- num
        // y_ext(t_curr) -- exact
        while (true) {
//            err_g.push_back(GLobal(y, t));
//            er.push_back((y[t.size()-1] - y_ext(t_curr)).norm());
            VectorXd y1 = RK_Step(k_coeff(t_curr, y[y.size()-1], h_curr), y[y.size()-1], h_curr);
            VectorXd y2 = RK_Step(k_coeff(t_curr + h_curr, y1, h_curr), y1, h_curr);
            VectorXd w = RK_Step(k_coeff(t_curr, y[y.size()-1], 2 * h_curr), y[y.size()-1], 2 * h_curr);
            double err = 1. / (pow(2, 4)- 1) * (y2-w).norm();
            double h_new = AutoStep(h_curr, tol, err, fac, facmax, facmin);
            if (err <= tol) {
                y.push_back(y2);
                // y_ext(t_curr);
                err_g.push_back(err);
                t_curr += 2. * h_curr;
                t.push_back(t_curr);
                h_curr = h_new;
                break;
            } else {
//                N++;
                h_curr = h_new;
                facmax = 1.;
            }
        }
    }
}


vector<VectorXd> k_coeffD (double t, VectorXd y, double h) {
    vector<vector<double>> a = {{}, {1./5.}, {3./40., 9./40.}, {44./45., -56./15., 32./9.}, {19372./6561., -25360./2187., 64448./6561., -212./729.}, {9017./3168., -355./33., 46732./5247., 49./176., -5103./18656.}, {35./384., 0., 500./1113., 125./192., -2187./6784., 11./84.}};
    vector<double> c = {0., 1./5., 3./10., 4./5., 8./9., 1., 1.};
    vector<VectorXd> k(7);
    VectorXd kv1(2);
    kv1(0) = f1(t, y);
    kv1(1) = f2(t, y);
    k[0] = kv1;
    
    VectorXd kv2(2);
    kv2(0) = f1(t + c[1] * h, y + h * a[1][0] * k[0]);
    kv2(1) = f2(t + c[1] * h, y + h * a[1][0] * k[0]);
    k[1] = kv2;
    
    VectorXd kv3(2);
    kv3(0) = f1(t + c[2] * h, y + h * (a[2][0] * k[0] + a[2][1] * k[1]));
    kv3(1) = f2(t + c[2] * h, y + h * (a[2][0] * k[0] + a[2][1] * k[1]));
    k[2] = kv3;
    
    VectorXd kv4(2);
    kv4(0) = f1(t + c[3] * h, y + h * (a[3][0] * k[0] + a[3][1] * k[1] + a[3][2] * k[2]));
    kv4(1) = f2(t + c[3] * h, y + h * (a[3][0] * k[0] + a[3][1] * k[1] + a[3][2] * k[2]));
    k[3] = kv4;
    
    VectorXd kv5(2);
    kv5(0) = f1(t + c[4] * h, y + h * (a[4][0] * k[0] + a[4][1] * k[1] + a[4][2] * k[2] + a[4][3] * k[3]));
    kv5(1) = f2(t + c[4] * h, y + h * (a[4][0] * k[0] + a[4][1] * k[1] + a[4][2] * k[2] + a[4][3] * k[3]));
    k[4] = kv5;
    
    VectorXd kv6(2);
    kv6(0) = f1(t + c[5] * h, y + h * (a[5][0] * k[0] + a[5][1] * k[1] + a[5][2] * k[2] + a[5][3] * k[3] + a[5][4] * k[4]));
    kv6(1) = f2(t + c[5] * h, y + h * (a[5][0] * k[0] + a[5][1] * k[1] + a[5][2] * k[2] + a[5][3] * k[3] + a[5][4] * k[4]));
    k[5] = kv6;
    
    VectorXd kv7(2);
    kv7(0) = f1(t + c[6] * h, y + h * (a[6][0] * k[0] + a[6][1] * k[1] + a[6][2] * k[2] + a[6][3] * k[3] + a[6][4] * k[4] + a[6][5] * k[5]));
    kv7(1) = f2(t + c[6] * h, y + h * (a[6][0] * k[0] + a[6][1] * k[1] + a[6][2] * k[2] + a[6][3] * k[3] + a[6][4] * k[4] + a[6][5] * k[5]));
    k[6] = kv7;
    
    return k;
}


VectorXd DP_Step1 (const vector<VectorXd>& k, const VectorXd& y, double h) {
    vector<double> b = {35./384., 0., 500./1113., 125./192., -2187./6784., 11./84., 0.};
    VectorXd y1 = y + h * (b[0] * k[0] + b[1] * k[1] + b[2] * k[2] + b[3] * k[3] + b[4] * k[4] + b[5] * k[5] + b[6] * k[6]);

    return y1;
}


VectorXd DP_Step2 (const vector<VectorXd>& k, const VectorXd& y, double h) {
    vector<double> b = {5179./57600., 0., 7571./16695., 393./640., -92097./339200., 187./2100., 1./40.};
    VectorXd y1 = y + h * (b[0] * k[0] + b[1] * k[1] + b[2] * k[2] + b[3] * k[3] + b[4] * k[4] + b[5] * k[5] + b[6] * k[6]);

    return y1;
}


void DormPrince (vector<VectorXd>& y, vector<double>& t, double facmax, double facmin, double fac, double tol, int N_Steps, vector<double>& err_g) {
    double Nv = 5.0;
    
    double t1 = Nv * 0.1;
    double t2 = t1 + 4.0;
    t.push_back(t1);

    double y10 = exp(sin(t1 * t1));
    double y20 = exp(cos(t1 * t1));
    VectorXd y0 {{y10, y20}};
    y.push_back(y0);

    double t_curr = t1;
    double h_curr = 0.1;
//    double facmax = 1.5;
    
//    vector<VectorXd> yxct;
//    yxct.push_back(y[0]);
    
    while (t_curr <= t2) {
        // Global
        // y[y.size() - 1] -- num
        // y_ext(t_curr) -- exact
        
        while (true) {
            VectorXd y1 = DP_Step1(k_coeffD(t_curr, y[y.size()-1], h_curr), y[y.size()-1], h_curr);
            VectorXd y2 = DP_Step2(k_coeffD(t_curr, y[y.size()-1], h_curr), y[y.size()-1], h_curr);
            double err = (y2-y1).norm();
//            err_g.push_back(GLobal(y, t));
            double h_new = AutoStep(h_curr, tol, err, fac, facmax, facmin);
            if (err <= tol) {
                y.push_back(y1);
                err_g.push_back(err);
//                yxct.push_back(y_ext(t_curr));
                t_curr += h_curr;
                t.push_back(t_curr);
                h_curr = h_new;
                break;
            } else {
                h_curr = h_new;
//                facmax = 1.0;
            }
        }
    }
    
//    cout << y.size() << endl;
//    cout << yxct.size() << endl;
//    cout << GLobal(y, yxct) << endl;
}

int main(int argc, const char * argv[]) {
    
    vector<VectorXd> y;
    vector<double> t;
    double tol = 0.0001;
    double fac = 0.8;
    double facmax = 1.5;
    double facmin = 0.5;
    
//--------------------------------------------------
//    Numerical solution/time
//--------------------------------------------------
//    const string path1 = "RungSol1.dat";
//    ofstream output1(path1);
//    const string path2 = "RungSol2.dat";
//    ofstream output2(path2);
//    const string path1 = "DormPrincSol1.dat";
//    ofstream output1(path1);
//    const string path2 = "DormPrincSol2.dat";
//    ofstream output2(path2);
    
//    int N = 0;
//    double f = 1.5;
//    DormPrince(y, t, f, N);
//    RungeKutta(y, t);
//
//    for (int i = 0; i < t.size()-1; ++i) {
//        output1 << t[i] << ',' << y[i](0) << endl;
//        output2 << t[i] << ',' << y[i](1) << endl;
//    }
    
//--------------------------------------------------
    
//--------------------------------------------------
//    Facmax tests
//--------------------------------------------------
//    const string path = "facmaxtime.dat";
//    ofstream output(path);
//    for (double f = 1.5; f <= 5.0; f += 0.5) {
//        int N = 0;
//        auto start = steady_clock::now();
//        DormPrince(y, t, f, N);
//        auto stop = steady_clock::now();
//        //    cout << duration_cast<milliseconds>(stop - start).count() << endl;
//        output << f << ',' << duration_cast<milliseconds>(stop - start).count() << endl;
//    }
//--------------------------------------------------

    
//--------------------------------------------------
//    t0 tests
//--------------------------------------------------
//    const string path = "t0.dat";
//    ofstream output(path);
//    for (double f = 0.05; f <= 0.5; f += 0.05) {
//        int N = 0;
//        auto start = steady_clock::now();
//        DormPrince(y, t, f, N);
//        auto stop = steady_clock::now();
//        cout << N << endl;
//        output << f << ',' << N << endl;
//        output << f << ',' << duration_cast<milliseconds>(stop - start).count() << endl;
//    }
//--------------------------------------------------

//    int N = 0;
//    auto start = steady_clock::now();
//    DormPrince(y, t, 1.5, N);
//    auto stop = steady_clock::now();
//    cout << duration_cast<milliseconds>(stop - start).count() << endl;
//    cout << N << endl;
//    int N = 0;
//    RungeKutta(y, t, N);
//    DormPrince(y, t, 1.5, N);
//    cout << GLobal(y, t) << endl;
  
//--------------------------------------------------
//    Local error
//--------------------------------------------------
//    int N = 0;
//    vector<double> er;
//    const string path = "localRt.dat";
//    ofstream output(path);
//    RungeKutta(y, t, facmax, facmin, fac, tol, N, er);
//    DormPrince(y, t, facmax, facmin, fac, tol, N, er);
//    for (int i = 0; i < t.size()-1; ++i) {
//        output << t[i] << ',' << er[i] << endl;
//    }
//--------------------------------------------------

//    for (double fac = 0.1; fac <= 0.7; fac += 0.02) {
//        DormPrince(y, t, 1.5, N, er, fac);
//        N.push_back(fac);
//    }
//    for (int i = 0; i < N.size()-1; ++i) {
//        cout << N[i] << ',' << er[i] << endl;
//        output << N[i] << ',' << er[i] << endl;
//    }
    
    return 0;
}
