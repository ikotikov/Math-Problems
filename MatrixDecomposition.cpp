//
//  main.cpp
//  MtrxDec
//
//  Created by Ivan Kotikov on 13.12.2021.
//

#include <iostream>
#include <Eigen/Dense>
#include <math.h>

using namespace std;
using namespace Eigen;

MatrixXd InitializeMatrix (int n, double eps) {
    MatrixXd A0(n, n);
    A0.setZero();
    for (auto i = 0; i < n; ++i) {
        A0(i, i) = 2;
        if (i < n - 1) {
            A0(i, i + 1) = -1;
            A0(i + 1, i) = -1;
        }
    }
    
    double C = 6.0/7.0 * eps;
    
    MatrixXd dA(n, n);
    dA.setZero();
    for (auto i = 0; i < n; ++i) {
        for (auto j = 0; j < n; ++j) {
            if (i != j) {
                dA(i,j) = C/(i + j);
            }
        }
    }
    MatrixXd A(n, n);
    A = A0 + dA;
    
    return A;
}

MatrixXd Qconstruct (const MatrixXd& A, const int& n, const int& i, const int& j, double q = 0.0) {
    MatrixXd Q(n, n);
    Q.setIdentity();
    double sq = sqrt((A(i, j) - q)*(A(i, j) - q) + A(i + 1, j)*A(i + 1, j));
    double c = (A(i, j) - q)/sq;
    double s = -A(i + 1, j)/sq;
    
    Q(i, i) = c;
    Q(i, i + 1) = -s;
    Q(i + 1, i) = s;
    Q(i + 1, i + 1) = c;
    
    return Q;
}

MatrixXd QHconstruct (const MatrixXd& A, const int& n, const int& i, const int& j, double q = 0.0) {
    MatrixXd Q(n, n);
    Q.setIdentity();
    double sq = sqrt((A(i, j) - q)*(A(i, j) - q) + A(i + 1, j)*A(i + 1, j));
    double c = (A(i, j) - q)/sq;
    double s = -A(i + 1, j)/sq;
    
    Q(i, i) = c;
    Q(i, i + 1) = s;
    Q(i + 1, i) = -s;
    Q(i + 1, i + 1) = c;
    
    return Q;
}

MatrixXd QRdecomposition (MatrixXd& R, VectorXd& b, const int& n) {
    MatrixXd Q(n, n);
    for (auto j = 0; j < n - 1; ++j) {
        for (auto i = n - 2; i >= j; --i) {
            Q = Qconstruct(R, n, i, j);
            R = Q * R;
            b = Q * b;
        }
    }
    return R;
}

VectorXd GaussBack (MatrixXd& R,const VectorXd& b, const int& n) {
    R.conservativeResize(n, n);
    int m = n - 1;
    R.col(m) = b;
    VectorXd x(m);
    for (auto i = m - 1; i >= 0; --i) {
        double s = 0.0;
        for(auto j = i + 1; j < m; ++j) {
            s += R(i, j) * x(j);
        }
        x(i) = (R(i, m) - s) / R(i, i);
    }
    return x;
}

MatrixXd Hessenberg (const MatrixXd& A, const int& n) {
    MatrixXd Q(n, n);
    MatrixXd H = A;
    for (auto j = 0; j < n - 2; ++j) {
        for (auto i = n - 2; i >= j + 1; --i) {
            Q = Qconstruct(A, n, i, j);
            H = Q * H * Q.transpose();
        }
    }
    return H;

}

VectorXd EigenV (const MatrixXd& H, int n) {
    MatrixXd A = H;
    MatrixXd Q(n, n);
    VectorXd EigenVals(n);
    EigenVals.setZero();
    VectorX<int> iterations(n);
    iterations.setZero();
    for (auto m = n; m >= 2; --m) {
        A.conservativeResize(m, m);
        while (abs(A(m - 1, m - 2)) > 1e-10) {
            double shift = A(m - 1, m - 1);
            for (auto j = 0; j < m - 1; ++j) {
                if (j == 0) {
                    Q = QHconstruct(A, m,  j, j, shift);
                    A = Q.transpose() * A * Q;
                } else {
                    Q = QHconstruct(A, m, j, j-1);
                    A = Q.transpose() * A * Q;
                }
            }
            ++iterations(m-1);
        }
        EigenVals(m - 1) = A(m - 1, m - 1);
    }
    EigenVals(0) = A(0, 0);
    cout << iterations << endl;
    sort(begin(EigenVals), end(EigenVals));
    return EigenVals;
}

MatrixXd EigVectors (const MatrixXd& A, const VectorXd& EigVal) {
    size_t n = A.diagonalSize();
    MatrixXd EigVecs(n, n);
    MatrixXd E(n, n), Ai(n, n);
    VectorXd xo(n), y(n);
    xo.setZero();
    y.setZero();
    for (int i = 0; i < n; ++i) {
        E.setIdentity();
        Ai = (A - (EigVal(i) + 1e-6) * E).inverse();
        VectorXd x {{6.0, 1.2, 0.19, 2.3, 1.9, 7.0, 8.0, 5.0, 2.0, 4.6}};
        while (fabs(1 - abs(x.transpose() * xo)) > 1e-15) {
            y = Ai * x;
            xo = x;
            x = y/y.norm();
        }
        EigVecs.row(i) = x;
    }
    return EigVecs;
}


VectorXd RealEigenVals (const int& n) {
    VectorXd EigenValues(n);
    for (auto i = 0; i < n; ++i) {
        EigenValues(i) = 2 * (1 - cos(M_PI * (i + 1) / (n + 1)));
    }
    return EigenValues;
}

MatrixXd RealEigenVectors (const int& n) {
    MatrixXd EigenVecs(n, n);
    for (auto j = 0; j < n; ++j) {
        for (auto k = 0; k < n; ++k) {
            EigenVecs(j, k) = sqrt(2. / (n + 1)) * sin(M_PI * (j + 1) * (k + 1) / (n + 1));
        }
    }
    return EigenVecs;
}



int main(int argc, const char * argv[]) {
    
    int n = 10;
    double eps = 1e-10;
    
    MatrixXd A = InitializeMatrix(n, eps);
//    MatrixXd R = A;
//    R.conservativeResize(n, n - 1);
//
//    VectorXd x0 {{6.0, 1.2, 0.19, 2.3, 1.9, 7.0, 8.0, 5.0, 2.0}};
//
//    VectorXd b(n);
//
//    b = R * x0;

//    QRdecomposition(R, b, n);

//    cout << "Residual = " << (GaussBack(R, b, n) - x0).norm()/x0.norm() << endl;

//    for (int i = 0; i < n; ++i) {
//        cout << "|l_" << i + 1 << " - l_0" << i + 1 << "| = " << abs(EigenV(Hessenberg(A, n), n)(i) - RealEigenVals(n)(i)) << endl;
//    }
    
    EigenV(Hessenberg(A, n), n);
    
//    cout << endl;
    
//    for (int i = 0; i < n; ++i) {
//        cout << "||z_" << i + 1 << " - z_0" << i + 1 << "|| = ";
//        if ((EigVectors(A, EigenV(Hessenberg(A, n), n)).row(i) - RealEigenVectors(n).row(i)).norm() > 1) {
//            cout << (EigVectors(A, EigenV(Hessenberg(A, n), n)).row(i) + RealEigenVectors(n).row(i)).norm() << endl;
//        } else {
//            cout << (EigVectors(A, EigenV(Hessenberg(A, n), n)).row(i) - RealEigenVectors(n).row(i)).norm() << endl;
//        }
//    }

    return 0;
}
