;(manage 'new 'deep-learning)

#|
Takes scalar and return 0.0 if negative otherwise it is the identity function
|#
(define (rectify-0 scalar)
  (cond
   ((<-0-0 scalar 0.0) 0.0)
   (else scalar)))

#|
< is not overwritten so <-0-0 is needed, this is just a note that if this ever becomes an issue to overwrite <
Tests:
(rectify-0 4)
;Value: 4

(rectify-0 -1)
;Value: 0.

|#

;;; extends rectify to operate on tensors of arbitrary size
;;; operates on the scalars of the tensor
(define rectify
  (ext1 rectify-0 0))

#|
Tests:
(rectify #(0 0.1 -0.1 3 -2))
;Value: #(0 .1 0. 3 0.)

(rectify #(0 -2 1 -1 1))
;Value: #(0 0. 1 0. 1)

|#

;;; Takes a tensor t and a theta (with weights and bias)
;;; and applies a linear operation of weights to the tensor t
;;; via the dot product and addition of the bias
;;; (ref theta 0) and t must be tensors of rank 1
(define linear-1-1
  (lambda (t)
    (guarantee (correct-rank-tensor? 1) t linear-1-1)
    (lambda (theta)
      (guarantee (correct-rank-tensor? 1) (ref theta 0))
      (+ (dot-product (ref theta 0) t) (ref theta 1)))))

#|
Tests:
(numerize ((linear-1-1 #(2 3)) (list #(-1 1) #(0.1))))
;Value: #(1.1)

(numerize ((linear-1-1 #(2 3)) (list #(1 -1) #(0.1))))
;Value #(-.9)

|#

;; applies relu on tensor 1 output after applying linear operation
(define relu-1-1
  (lambda (t)
    (lambda (theta)
      (rectify ((linear-1-1 t) theta)))))

#|
Tests:
(numerize ((relu-1-1 #(2 3)) (list #(-1 1) #(0.1))))
;Value: #(1.1)

(numerize ((relu-1-1 #(2 3)) (list #(1 -1) #(0.1))))
;Value: #(0.)

|#

#|theta-0 is the weights
theta-1 is the bias|#

#|
More Tests:

((relu-1-1 (tensor 2.0 1.0 3.0))
 (list (tensor 7.1 4.3 -6.4) 0.6))      ;Value: 0.0

(numerize (rectify
	   (+
	    (dot-product (tensor 7.1 4.3 -6.4) (tensor 2.0 1.0 3.0))
	    0.6))
	  ) ;Value: 0.0

((relu-1-1 (tensor 0.5))
 (list (tensor 1.0) -1.0)) ;Value: 0.0

#|0.5 is an x coord for this relu graph and the result 0.0 is the y coord|#
|#

#|
Sample Little Learner code (not needed):

(define half-strip
  (lambda (x theta)
    (- ((relu-1-1 (tensor x)) (list (ref theta 0) (ref theta 1)))
       ((relu-1-1 (tensor x)) (list (ref theta 0) (ref theta 2))))))

(numerize (half-strip 1.25 (list (tensor 1.0) -1.0 -1.5))) ;Value: 0.25

(define full-strip
  (lambda (x theta)
    (- (half-strip x (list (ref theta 0) (ref theta 1) (ref theta 2)))
       (half-strip x (list (ref theta 3) (ref theta 4) (ref theta 5))))))

(numerize (full-strip 3.25 (list (tensor 1.0)
				 -1.0
				 -1.5
				 (tensor 1.0)
				 -3.0
				 -3.5))) ;Value: .25
(define (test-graph x)
  (+ (full-strip x
		 (list (tensor 1.0)
		       -1.0
		       -1.5
		       (tensor 1.0)
		       -3.0 -3.5))
     (half-strip x
		 (list (tensor 1.0)
		       -1.0
		       -1.5))))

(numerize (test-graph 3.5)
	  ) ;Value: 0.5

(numerize (test-graph 1.5)
	  ) ;Value: 1.0

|#

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;;Layers
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

#|
Layer function that makes a 4 neuron layer of Relu's

(lambda (t)
  (lambda (theta)
    (let ((w (ref theta 0)) (b (ref theta 1)))
      (tensor ((relu-1-1 t) (list (ref w 0) (ref b 0)))
	      ((relu-1-1 t) (list (ref w 1) (ref b 1)))
	      ((relu-1-1 t) (list (ref w 2) (ref b 2)))
	      ((relu-1-1 t) (list (ref w 3) (ref b 3))))
      )))

Law of Dense Layers: A dense layer function invokes m neurons on an n-element input tensor that produces an m-element output tensor
'The Little Learner' Chapter 11, pg. 218	       
|#

#|
if w is a tensor of shape (m n) and t is of shape (n) the output will be shape (m)
This is the matrix vector multiplication operation where w is a matrix and t is a
column vector
|#
(define dot-product-2-1
  (lambda (w t)
    (sum
     (*-2-1 w t))))

#|
Tests:
(numerize (dot-product-2-1 #(#(3 2 1) #(-3 -2 -1) #(6 7 8)) #(0 0 1)))
;Value: #(1 -1 8)

(numerize (dot-product-2-1 #(#(3 2 1) #(-3 -2 -1) #(6 7 8)) #(0 1 0)))
;Value: #(2 -2 7)

(numerize (dot-product-2-1 #(#(3 2 1) #(-3 -2 -1) #(6 7 8)) #(1 0 0)))
;Value: #(3 -3 6)

(numerize (dot-product-2-1
	       (tensor (tensor 2.0 1.0 3.1) (tensor 3.7 4.0 6.1))
	       (tensor 1.3 0.4 3.3))
	      )                             ;Value: #(13.23 26.54)

|#

#|t is shape (n) and theta-0 is shape (m n) theta-1 is shape (m) then ((linear t) theta) is shape (m)
linear applies matrix-vector multiplication with the weights matrix and a tensor. Then adds
the bias vector. Essentially this multiplies out one layer of inputs with its weights and adds
a bias vector.
|#
(define linear
  (lambda (t)
    (lambda (theta)
      (+ (dot-product-2-1 (ref theta 0) t) (ref theta 1)))))

#|
Tests:
(numerize ((linear #(0 1 1)) (list #(#(-1 1 4) #(-1 2 2) #(-1 3 1) #(-1 4 3)) #(100 200 300 400))))
;Value: #(105 204 304 407)

|#

#|relu is ame as relu-1-1 but it uses linear instead of linear-1-1
This applies the relu function to the output of the linear operation Wt + b
Where W is a matrix of weights, t is a column vector and b is a column vector
|#

(define relu
  (lambda (t)
    (lambda (theta)
      (rectify ((linear t) theta)))))

#|
Suppose input tensor is shape (4) and we want to pass it through a layer of 3 neurons so output is shape (3), then theta-0 is shape (3 4) and theta-1 shape of (3)
|#

#|
Network function
assembles layer functions together so that the output of one layer becomes the input to the next layer
|#

(define 1-relu
  (lambda (t)
    (lambda (theta)
      ((relu t) theta))))

(define 2-relu
  (lambda (t)
    (lambda (theta)
      ((relu
	((relu t) theta))
       (drop theta 2)))))

(define 3-relu
  (lambda (t)
    (lambda (theta)
      ((2-relu
	((relu t) theta))
       (drop theta 2)))))

#|General version of above functions that can make k layers|#

(define k-relu
  (lambda (k)
    (lambda (t)
      (lambda (theta)
	(cond
	 ((zero? k) t)
	 (else (((k-relu (sub1 k))
		 ((relu t) theta)
		 (drop theta 2)))))))))
#|
Ex inputs and layers
input of shape (32)
first layer is 64 neurons
theta-0 is shape (64 32)
theta-1 shape (64)
input into second layer is shape (64)
second layer is 45 neurons
theta-2 shape is (45 64)
theta-3 is shape (45)
input to third layer is shape (45)
third layer is 26 neurons
theta-4 is shape (26 45)
theta-5 is shape (26)
|#


