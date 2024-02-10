import os
import cv2
import numpy as np
import open3d as o3d

def main(folder_path):
    # Listar todos os arquivos na pasta
    files = os.listdir(folder_path)

    # Verificar se há imagens na pasta
    if len(files) == 0:
        print('Nenhuma imagem válida encontrada na pasta.')
        return

    # Lista para armazenar as imagens
    images = []

    # Carregar cada imagem
    for filename in files:
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Caminho completo para a imagem
            image_path = os.path.join(folder_path, filename)

            # Carregar a imagem
            img = cv2.imread(image_path)

            # Verificar se a imagem foi carregada corretamente
            if img is None:
                print(f'Erro ao carregar a imagem {filename}.')
                continue

            images.append(img)

    # Calcular correspondências entre as imagens
    matches_list = []
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for i in range(len(images) - 1):
        keypoints1, descriptors1 = orb.detectAndCompute(images[i], None)
        keypoints2, descriptors2 = orb.detectAndCompute(images[i + 1], None)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        matches_list.append(matches)

    # Calcular a matriz fundamental usando as correspondências
    K = np.array([[800, 0, 525], [0, 600, 525], [0, 0, 1]])  # Matriz de calibração intrínseca
    E_list = []
    for i, matches in enumerate(matches_list):
        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        E, _ = cv2.findEssentialMat(pts1, pts2, K)
        E_list.append(E)

    # Calcular a nuvem de pontos
    pcd = o3d.geometry.PointCloud()

    for i, E in enumerate(E_list):
        R1, R2, t = cv2.decomposeEssentialMat(E)
        R = R1 if np.linalg.det(R1) > 0 else R2
        T, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)
        extrinsic = np.hstack((R, t))
        extrinsic = np.vstack((extrinsic, [0, 0, 0, 1]))
        intrinsic = o3d.camera.PinholeCameraIntrinsic(800, 600, 525, 525, 400, 300)
        view = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
        pcd.points.extend(o3d.utility.Vector3dVector(T @ intrinsic.intrinsic_matrix @ pts1))  # Transformando pontos para o sistema de coordenadas global
        pcd.colors.extend(o3d.utility.Vector3dVector(images[i][:, :, ::-1] / 255))  # Convertendo de BGR para RGB e normalizando para [0, 1]

    # Visualizar a nuvem de pontos
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    folder_path = r'C:\Users\Jhona\Desktop\IMAGENS_DRONE'  # Caminho para a pasta contendo as imagens do drone
    main(folder_path)
