__authors__ = ['1709992', '1711342', '1620854', '1641014']
__group__ = '13'

import numpy as np
import matplotlib.pyplot as plt
from time import time
from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_retrieval
from KNN import KNN
from Kmeans import KMeans, get_colors
from threading import Thread
from os import cpu_count
from os.path import isfile
import pickle


if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    def add_color(idx:int, stored:list[list], imgs, k, options):
        for img in imgs:
            km = KMeans(img, k, options)
            km.find_bestK(k)
            km.fit()
            colors = get_colors(km.centroids)
            stored[idx].append(colors)


    def options_dialog() -> dict:
        options = {}
        customize = input("Do you want to costumize Kmeans opptions? [Y/n] ")
        if not customize.lower() == "y":
            return options

        print("""Choose a 'km_init':
    1: first
    2: naive shearding
    3: k-means++
    4: custom
    5: random""")
        opt = input("Option: [1,2,3,4,5] ")
        if opt not in ["1", "2", "3", "4", "5"]:
            print("Unrecognised option defaulting to first")
            opt = "1"
        options["km_init"] = ["first", "naive shearding", "k-means++", "custom", "random"][int(opt)]
        

        print("""Choose a 'fitting':
    1: WCD
    2: interClassDistance
    3: intraClassDistance
    4: fisher""")
        opt = input("Option: [1, 2, 3, 4] ")
        if opt not in ["1", "2", "3", "4"]:
            print("Unrecognised option defaulting to WCD")
            opt = "1"
        options["fitting"] = ["WCD", "interClassDistance", "intraClassDistance", "fisher"][int(opt)]

        return options



    def get_labels(k: int = 1):
        if isfile("labels.pkl"):
            if input("Load saved labels? [Y/n] ").lower() == "y":
                with open("labels.pkl", "rb") as f:
                    shape_labels, color_labels = pickle.load(f)
                return shape_labels, color_labels

        options = options_dialog()

        knn = KNN(train_imgs, train_class_labels)
        shape_labels = knn.predict(test_imgs, k)
        color_labels = []

        threads = []
        stored = [[] for _ in range(cpu_count()+1)]

        img_per_thread = len(test_imgs)//cpu_count()

        for i in range(cpu_count()):
            threads.append(Thread(target=add_color, args=(i, stored, test_imgs[i*img_per_thread:(i+1)*img_per_thread], k, options)))
        threads.append(Thread(target=add_color, args=(cpu_count(), stored, test_imgs[cpu_count()*img_per_thread:len(test_imgs)], k, options)))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()
        color_labels = sum(stored, [])

        with open("labels.pkl", "wb") as f:
            pickle.dump((shape_labels, color_labels), f)

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
            temps.append(time()-start)
            iteracions.append(kmeans.num_iter)
            wcds.append(kmeans.withinClassDistance())
        return wcds, iteracions, temps

    def get_shape_accuracy(labels, ground_truth):
        if not len(labels) == len(ground_truth):
            raise ValueError("labels and ground truth should be the same size")

        correct = np.equal(labels, ground_truth)
        accuracy = np.count_nonzero(correct) / len(labels)

        return accuracy

    def get_color_accuracy(labels, ground_truth):
        if not len(labels) == len(ground_truth):
            raise ValueError("labels and ground truth should be the same size")

        accuracy_total = 0
        for label, gt in zip(labels, ground_truth):
            accuracy_total += len(set(gt).intersection(set(label)))/len(set(gt))

        accuracy = accuracy_total / len(labels)

        return accuracy
    

    def visualize_kmeans_statistics(wcds, iteracions, temps, shape_accuracy, color_accuracy):
        plt.figure(figsize=(10, 6))

        plt.suptitle(f"Color accuracy: {color_accuracy*100}%   Shape accuracy: {shape_accuracy*100}%")


        plt.subplot(1, 3, 1)
        plt.plot(range(2, len(wcds)+2), wcds, marker="o")
        plt.title("Distancia interclass (withinClassDistance)")
        plt.xlabel("Numero de clusters (K)")
        plt.ylabel("WCD")

        plt.subplot(1, 3, 2)
        plt.plot(range(2, len(iteracions)+2), iteracions, marker="o")
        plt.title("Numero d'iteracions")
        plt.xlabel("Numero de clusters (K)")
        plt.ylabel("Iteracions")

        plt.subplot(1, 3, 3)
        plt.plot(range(2, len(temps)+2), temps, marker="o")
        plt.title("Temps de convergencia")
        plt.xlabel("Numero de clusters (K)")
        plt.ylabel("Temps")

        plt.tight_layout()
        plt.show()

    def main():
        print("""Select test method:
    1: Retrieval by colour (KMeans)
    2: Retrieval by shape (KNN)
    3: Retrieval by colour and shape (KMeans and KNN)
    4: Quantitative analysis""")
        n = input("Method seletion: ")
        n_elem = 20
        if n not in ["1", "2", "3", "4"]:
            print("Method not recognised, exiting...")
            return
        shape_labels, color_labels = get_labels(10)
        match int(n):
            case 1:
                my_querry = input("Choose colors to querry [coma separeated ex: \"pink,red\"]: ")
                my_querry = my_querry.split(",")
                idxs, imgs = retrieval_by_color(test_imgs, color_labels, my_querry)
                visualize_retrieval(imgs, n_elem, info=[test_color_labels[i] for i in idxs], ok=[any(query in np.char.lower(test_color_labels[i]) for query in np.char.lower(my_querry)) for i in idxs], title=my_querry)

            case 2:
                my_querry = input("Choose a shape to querry: ")
                idxs, imgs = retrieval_by_shape(test_imgs, shape_labels, my_querry)
                print(idxs, imgs)
                visualize_retrieval(imgs, n_elem, info=[test_class_labels[i] for i in idxs], ok=[my_querry.lower() == test_class_labels[i].lower() for i in idxs], title=my_querry)

            case 3:
                shape_querry = input("Choose a shape to querry: ")
                color_querry = input("Choose colors to querry [coma separeated ex: \"pink,red\"]: ")
                color_querry = color_querry.split(",")
                idxs, imgs = retrieval_combined(test_imgs, shape_labels, color_labels, shape_querry, color_querry)
                visualize_retrieval(imgs, n_elem, info=[(test_color_labels[i], test_class_labels[i]) for i in idxs], ok=[any(query in np.char.lower(test_color_labels[i]) for query in np.char.lower(color_querry)) and shape_querry.lower() == test_class_labels[i].lower() for i in idxs], title=f"{', '.join(color_querry)}, {shape_querry}")

            case 4:
                visualize_kmeans_statistics(*kmean_statistics(KMeans(test_imgs[0]), 10), get_shape_accuracy(shape_labels, test_class_labels), get_color_accuracy(color_labels, test_color_labels))

    main()
