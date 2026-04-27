import numpy as np
import cv2
from scipy.spatial.distance import cdist

class FeatureMatcher:
    def __init__(self, ratio_threshold=0.75, distance_metric='euclidean'):
        """
        Initialize feature matcher.
        
        Args:
            ratio_threshold (float): Threshold for Lowe's ratio test
            distance_metric (str): Distance metric for descriptor matching
        """
        self.ratio_threshold = ratio_threshold
        self.distance_metric = distance_metric
    
    def match_descriptors(self, desc1, desc2):
        """
        Match descriptors using Lowe's ratio test.
        
        Args:
            desc1 (numpy.ndarray): First set of descriptors
            desc2 (numpy.ndarray): Second set of descriptors
            
        Returns:
            list: List of DMatch objects
        """
        distances = cdist(desc1, desc2, metric=self.distance_metric)

        matches = []
        for i in range(distances.shape[0]):
            idx = np.argsort(distances[i])
            j1 = idx[0]
            j2 = idx[1]

            d1 = distances[i, j1]
            d2 = distances[i, j2]

            if d1 < self.ratio_threshold * d2:
                matches.append(cv2.DMatch(i, int(j1), float(d1)))

        return matches

class RANSAC:
    def __init__(self, n_iterations=1000, inlier_threshold=3.0, min_inliers=5):
        """
        Initialize RANSAC algorithm for homography estimation.
        
        Args:
            n_iterations (int): Number of RANSAC iterations
            inlier_threshold (float): Threshold for inlier identification
            min_inliers (int): Minimum number of inliers for a valid model
        """
        self.n_iterations = n_iterations
        self.inlier_threshold = inlier_threshold
        self.min_inliers = min_inliers
    
    def estimate_homography(self, src_points, dst_points):
        """
        Estimate homography matrix using RANSAC.
        
        Args:
            src_points (numpy.ndarray): Source points (N, 2)
            dst_points (numpy.ndarray): Destination points (N, 2)
            
        Returns:
            tuple: (H, inliers) where H is the homography matrix and
                  inliers is a binary mask of inlier matches
        """
        assert src_points.shape[0] == dst_points.shape[0], "Number of points must match"
        assert src_points.shape[0] >= 4, "At least 4 point pairs are required"
        
        # Implement RANSAC algorithm for homography estimation
        # HINT: 1. Randomly select 4 point pairs
        n_points = src_points.shape[0]
        best_H = None
        best_inliers = None
        best_count = 0
        for i in range(self.n_iterations):
            idx = np.random.choice(n_points, 4, replace = False)
            src_sample = src_points[idx]
            dst_sample = dst_points[idx]

            #       2. Compute homography
            H = cv2.getPerspectiveTransform(src_sample.astype(np.float32), dst_sample.astype(np.float32))

            #       3. Transform all points
            src_reshaped = src_points.reshape(-1, 1, 2).astype(np.float32)
            projected = cv2.perspectiveTransform(src_reshaped, H).reshape(-1,2)

            #       4. Identify inliers
            errors = np.linalg.norm(projected - dst_points, axis = 1)
            inliers = errors < self.inlier_threshold

            #       5. Keep the best model
            count = int(inliers.sum())
            if count > best_count:
                best_count = count
                best_H = H
                best_inliers = inliers
        
        return best_H, best_inliers
    
    def compute_match_quality(self, H, src_points, dst_points, inliers):
        """
        Compute match quality score based on homography transformation.
        
        Args:
            H (numpy.ndarray): Homography matrix
            src_points (numpy.ndarray): Source points
            dst_points (numpy.ndarray): Destination points
            inliers (numpy.ndarray): Binary mask of inlier matches
            
        Returns:
            float: Match quality score
        """
        # TODO: Implement match quality evaluation
        # HINT: Consider inlier ratio and transformation error
        
        # Your implementation here
        ratio = inliers.sum() / len(inliers)
        projected = cv2.perspectiveTransform(src_points.reshape(-1,1,2).astype(np.float32), H).reshape(-1,2)
        errors = np.linalg.norm(projected - dst_points, axis=1)
        
        if inliers.sum() > 0:
            mean_error = errors[inliers].mean()
        else:
            mean_error = float('inf')

        quality_score = ratio / (1.0 + mean_error)
        
        return quality_score