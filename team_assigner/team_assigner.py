
from sklearn.cluster import KMeans



class Teamassigner:
    def __init__(self):
        self.team_colors = {}
        self.runner_team_dict = {}
        

    def get_clustering_model(self,image):
        # Reshape the image to 2 Darray

        image_2d = image.reshape(-1, 3) 

        # Preform k- means with 1 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1).fit(image_2d)

        return kmeans

    def get_runner_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        top_half_image = image[0:int(image.shape[0]/2), :]

        # Get clustering model
        
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape 
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Get the player cluster 
        corner_clusters = [clustered_image[0,0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_runner_cluster = max(set(corner_clusters), key=corner_clusters.count)
        runner_cluster = 1 - non_runner_cluster

        # Access the runner cluster center based on the number of clusters
        if len(kmeans.cluster_centers_) > 1:
            # If there are multiple clusters, use the runner cluster logic
            runner_color = kmeans.cluster_centers_[runner_cluster]  # runner_cluster would be defined based on your logic
        else:
            # If there is only one cluster, use the first one
            runner_color = kmeans.cluster_centers_[0]

        return runner_color


    def assign_team_color(self, frame, runner_detections):

        runner_colors = []

        for _, runner_detection in runner_detections.items():
            bbox = runner_detection["bbox"]
            runner_color = self.get_runner_color(frame, bbox)
            runner_colors.append(runner_color)


        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(runner_colors)

        self.kmeans = kmeans

        # Access the runner cluster center based on the number of clusters
        if len(kmeans.cluster_centers_) > 1:
            # If there are multiple clusters, use the runner cluster logic
            self.team_colors[1] = kmeans.cluster_centers_[0]
            self.team_colors[2] = kmeans.cluster_centers_[1]
        else:
            # If there is only one cluster, use the first one
            self.team_colors[1] = kmeans.cluster_centers_[0]
            


    def get_runner_team(self, frame, runner_bbox, runner_id):
        if runner_id in self.runner_team_dict:
            return self.runner_team_dict[runner_id]
        

        runner_color = self.get_runner_color(frame, runner_bbox)

        runner_id = self.kmeans.predict(runner_color.reshape(1, -1))[0]
        runner_id += 1

        self.runner_team_dict[runner_id] = runner_id

        return runner_id

        