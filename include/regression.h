#pragma once

#include "opencv2\opencv.hpp"
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_line.h>

double LinearRegression(std::vector<std::tuple<int, int>>& input)
{
	long i;
	double a = 0.0;
	double b = 0.0;
	double sumX = 0.0; double sumY = 0.0; double sumXsquared = 0.0; double sumYsquared = 0.0; double sumXY = 0.0;
	double coefD = 0.0; double coefC = 0.0; double stdError = 0.0;
	long n = 0L;
	std::vector<int> x;
	std::vector<int> y;
	for (int i = 0; i < input.size(); i++) {
		x.push_back(std::get<0>(input.at(i)));
		y.push_back(std::get<1>(input.at(i)));
	}

	if (input.size() > 0L) {// if size greater than zero there are data arrays
		for (n = 0, i = 0L; i < input.size(); i++) {
			n++;
			sumX += x[i];
			sumY += y[i];
			sumXsquared += x[i] * x[i];
			sumYsquared += y[i] * y[i];
			sumXY += x[i] * y[i];
			if (fabs(double(n) * sumXsquared - sumX * sumX) > DBL_EPSILON)
			{
				b = (double(n) * sumXY - sumY * sumX) /
					(double(n) * sumXsquared - sumX * sumX);
				a = (sumY - b * sumX) / double(n);

				double sx = b * (sumXY - sumX * sumY / double(n));
				double sy2 = sumYsquared - sumY * sumY / double(n);
				double sy = sy2 - sx;

				coefD = sx / sy2;
				coefC = sqrt(coefD);
				stdError = sqrt(sy / double(n - 2));
			}
			else
			{
				a = b = coefD = coefC = stdError = 0.0;
			}
		}
	}
	return stdError;
}

std::pair<Eigen::Vector3f, Eigen::Vector3f> fitLine(std::vector<Eigen::Vector3f>& points) {
	// copy coordinates to  matrix in Eigen format
	size_t num_atoms = points.size();
	Eigen::Matrix< Eigen::Vector3f::Scalar, Eigen::Dynamic, Eigen::Dynamic > centers(num_atoms, 3);
	for (size_t i = 0; i < num_atoms; ++i) centers.row(i) = points[i];

	Eigen::Vector3f origin = centers.colwise().mean();
	Eigen::MatrixXf centered = centers.rowwise() - origin.transpose();
	Eigen::MatrixXf cov = centered.adjoint() * centered;
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig(cov);
	Eigen::Vector3f axis = eig.eigenvectors().col(2).normalized();
	//multiply with -1 so that it points towards origin
	return std::make_pair(origin, axis * -1.0f);
}


// regress the variable t in the equation
// y = m * x + t
// when m is fixed
// for the given input values
double regress_t_with_fixed_m(std::vector<std::tuple<int, int>>& pos, double m)
{
	double n = pos.size();

	double accum = 0.0;
	for (int i = 0; i < n; i++)
	{
		accum += std::get<1>(pos[i]) - m * std::get<0>(pos[i]);
	}
	double error = 0.0;
	double t = accum / n;
	for (int j = 0; j < n; j++) {
		double tmp = (std::get<1>(pos[j]) - t) * (std::get<1>(pos[j]) - t);
		error += tmp;
	}

	return error / n;
}

double regress_split_at(std::vector<std::tuple<int, int>>& part_a, std::vector<std::tuple<int, int>>& part_b)
{
	double error_a = LinearRegression(part_a);
	double error_b = regress_t_with_fixed_m(part_b, 0.0);
	return error_a + error_b;
}

int regression(boost::shared_ptr<std::vector<std::tuple<int, int, cv::Mat, cv::Mat>>>& needle_width) {
	std::vector<std::tuple<int, int>> widths;
	for (int i = 0; i < needle_width->size(); i++) {
		widths.push_back(std::tuple<int, int>(std::get<0>(needle_width->at(i)), std::get<1>(needle_width->at(i))));
	}
	std::vector<double> errors;
	for (int j = 3; j < widths.size(); j++) {
		errors.push_back(regress_split_at(std::vector<std::tuple<int, int>>(widths.begin(), widths.begin() + j), std::vector<std::tuple<int, int>>(widths.begin() + j, widths.end())));
	}
	int error_min_index = 0;
	for (int k = 0; k < errors.size(); k++) {
		if (errors[k] < errors[error_min_index]) {
			error_min_index = k;
		}
	}
	int index = error_min_index;
	//add number of frames at which oct cloud starts
	return index + std::get<0>(widths.at(0));
}