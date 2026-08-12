from helper import load_data, show_images
from solution import PCA, reconstruct_error
import numpy as np

def test_pca():
	dataloc = "../data/USPS.mat"
	A = load_data(dataloc)
	ps = [10, 50, 100, 200]
	for p in ps:
		pca = PCA(A, p)
		Ap = pca.get_reduced()
		A_re = pca.reconstruction(Ap)
		error = reconstruct_error(A, A_re)
		print(error)
		show_images(A_re, p, 1)
		show_images(A_re, p, 2)

if __name__ == '__main__':
	test_pca()
