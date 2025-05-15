__authors__ = ['1709992', '1711342', '1620854', '1641014']
__group__ = '13'

import numpy as np
import matplotlib as plt
from time import time
from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_retrieval
from KNN import KNN
from Kmeans import KMeans, get_colors


if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    def get_labels(k: int = 1):
        #knn = KNN(train_imgs, train_class_labels)
        #shape_labels = knn.predict(test_imgs, k)
        shape_labels = []
        color_labels = []
        options = {}


        for img in test_imgs:
            km = KMeans(img, k, options)
            km.find_bestK(k)
            km.fit()
            colors = get_colors(km.centroids)
            color_labels.append(colors)

        return shape_labels, color_labels


    # You can start coding your functions here
    def retrieval_by_color(images, labels, querry_color) -> tuple[list[int], np.array]:
        if isinstance(querry_color, list):
            querry_color = [querry_color]

        ok_images = []
        idxs = []

        for i, (img, lbl) in enumerate(zip(images, labels)):
            if any([querry in np.char.lower(lbl) for querry in np.char.lower(querry_color)]):
                ok_images.append(img)
                idxs.append(i)
        return idxs, np.array(ok_images)

    def retrieval_by_shape(images, labels, querry_shape) -> tuple[list[int], np.array]:
        ok_images = []
        idxs = []

        for i, (img, lbl) in enumerate(zip(images, labels)):
            if querry_shape.lower() == lbl.lower():
                ok_images.append(img)
                idxs.append(i)
        return idxs, np.array(ok_images)

    def retrieval_combined(images, shape_labels, color_labels, querry_shape, querry_color) -> tuple[list[int], np.array]:
        ok_images = []
        idxs = []

        for i, (img, slbl, clbl) in enumerate(zip(images, shape_labels, color_labels)):
            if querry_shape.lower() == slbl.lower() and  any([querry in np.char.lower(clbl) for querry in np.char.lower(querry_color)]):
                ok_images.append(img)
                idxs.append(i)
    
        return idxs, np.array(ok_images)


    def kmean_statistics(kmeans:KMeans, kmax):
        iteracions = []
        wcds = []
        temps = []
        for k in range(2, kmax+1):
            kmeans.K = k
            start = time()
            kmeans.fit()
            time.append(time()-start)
            iteracions.append(kmeans.num_iter)
            wcds.append(kmeans.withinClassDistance())
        return wcds, iteracions, temps


    def visualize_kmeans_statistics(wcds, iteracions, temps):
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 3, 1)
        plt.plot(range(2, len(wcds)), wcds, marker="o")
        plt.title("Distancia interclass (withinClassDistance)")
        plt.xlabel("Numero de clusters (K)")
        plt.ylabel("WCD")

    def main():
        print("""Select test method:
    1: Retrieval by colour (KMeans)
    2: Retrieval by shape (KNN)
    3: Retrieval by colour and shape (KMeans and KNN)
    4: Quantitative analysis""")
        n = input("Method seletion: ")
        n_elem = 25
        shape_labels, color_labels = get_labels(10)
        if n not in ["1", "2", "3", "4"]:
            print("Method not recognised, exiting...")
            return

        match int(n):
            case 1:
                my_querry = input("Choosea color que querry: ")
                my_querry = my_querry.split(",")
                idxs, imgs = retrieval_by_color(test_imgs, color_labels, my_querry)
                visualize_retrieval(imgs, n_elem, info=[test_class_labels[i] for i in idxs], ok=[any(query in np.char.lower(test_color_labels[i]) for query in np.char.lower(my_querry)) for i in idxs], title=my_querry)

            

    main()
