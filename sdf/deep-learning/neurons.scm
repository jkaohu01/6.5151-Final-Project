;(manage 'new 'deep-learning)

#|
Takes scalar and return 0.0 if negative otherwise it is the identity function
|#

(define rectify-0
  (lambda (s)
    (cond
     ((<-0-0 s 0.0) 0.0)
     (else s))))

#|
< is not overwritten so <-0-0 is needed, this is just a note that if this ever becomes an issue to overwrite <
|#

(define rectify
  (ext1 rectify-0 0))

(define linear-1-1
  (lambda (t)
    (lambda (theta)
      (+ (dot-product (ref theta 0) t) (ref theta 1)))))

(define relu-1-1
  (lambda (t)
    (lambda (theta)
      (rectify ((linear-1-1 t) theta)))))

#|theta-0 is the weights
theta-1 is the bias|#

((relu-1-1 (tensor 2.0 1.0 3.0))
 (list (tensor 7.1 4.3 -6.4) 0.6));Value: 0.0

(numerize (rectify
	   (+
	    (dot-product (tensor 7.1 4.3 -6.4) (tensor 2.0 1.0 3.0))
	    0.6))
	  ) ;Value: 0.0

((relu-1-1 (tensor 0.5))
 (list (tensor 1.0) -1.0)) ;Value: 0.0

#|0.5 is an x coord for this relu graph and the result 0.0 is the y coord|#

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
|#

(define dot-product-2-1
  (lambda (w t)
    (sum
     (*-2-1 w t))))

(numerize (dot-product-2-1
	   (tensor (tensor 2.0 1.0 3.1) (tensor 3.7 4.0 6.1))
	   (tensor 1.3 0.4 3.3))
	  ) ;Value: (13.23 26.54)
#|t is shape (n) and theta-0 is shape (m n) theta-1 is shape (m) then ((linear t) theta) is shape (m)|#

(define linear
  (lambda (t)
    (lambda (theta)
      (+ (dot-product-2-1 (ref theta 0) t) (ref theta 1)))))


#|relu is ame as relu-1-1 but it uses linear instead of linear-1-1|#

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


