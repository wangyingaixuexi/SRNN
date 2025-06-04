import numpy as np
import open3d as o3d



def visualize_clusters(mesh: o3d.geometry.TriangleMesh):
    face_to_cluster_ID, n_faces, _ = mesh.cluster_connected_triangles()
    face_to_cluster_ID = np.asarray(face_to_cluster_ID)
    n_clusters = len(n_faces)
    clusters = [o3d.geometry.TriangleMesh() for i in range(n_clusters)]
    cluster_faces = [list() for i in range(n_clusters)]
    faces = np.asarray(mesh.triangles)
    for ID in range(n_clusters):
        clusters[ID].vertices = mesh.vertices
        clusters[ID].triangles = o3d.utility.Vector3iVector(
            faces[face_to_cluster_ID == ID]
        )
        clusters[ID].remove_unreferenced_vertices()
        clusters[ID].paint_uniform_color(np.random.random_sample((3,)))
    o3d.visualization.draw_geometries(clusters)
