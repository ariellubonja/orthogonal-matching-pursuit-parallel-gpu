too many changes to compare repos directly. 

Starting point: 3.9 working

1. new python version has different package resolution. 
	- 3.9 works for loading Cython
		from cython.test import * ✅ 		
	- 3.12 says module not found
		from test import * ✅
		
		
2. Dimension issue on 3.12:

Single core. v0 fast implementation.
/src/main.py", line 223, in omp_v0
    projections = (X.transpose(1, 0) @ y[:, :, None]).squeeze(-1)
                   ~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~
RuntimeError: Expected size for first two dimensions of batch2 tensor to be: [100, 100] but got: [100, 50].

	- X is the same shape both times: 100,100
	- py39: y[:, :, None] is (100,50) ✅
	- py312: y[:, :, None] is (50,100) 
	
	    y, X, w = make_sparse_coded_signal(
	    	- Y got transposed in newer Sklearn
	    	
	    	
argmax_blast not defined in py312 - import in #1 is failing silently

	- fixed by editing setup.py to work from project root
	
normalize keyword not found

	- removing it increases error py312
	
	
