;; A simple function for creating 2d lines
;; ---------------------------------------
;; x: an input number to a line function
(define (line x)
  ;; theta: the 2 parameters to the line: m, b in y = mx + b
  ;;        in list form
  ;; output: y = mx + b
  (lambda (theta)
    (+ (* (ref theta 0) x) (ref theta 1))))

#|
Tests:
(numerize ((line 7.3) (list 1.0 0)))
;Value: 7.3

(numerize ((line 2) (list 4 -1)))
Value: 7

|#

;; toy linear dataset
(define line-xs
  (tensor 2.0 1.0 4.0 3.0))

;; toy linear dataset
(define line-ys
  (tensor 1.8 1.2 4.2 3.3))

;; loss^2 for a given dataset
;; ---------------------------------
;; target: some function awaiting first data (xs ys)
;;         then weights (theta), it specifies how
;;         outputs are predicted given inputs (how
;;         some y output is predicted given an x feature)
(define l2-loss
  (lambda (target)
    ;; xs: features
    ;; ys: outputs
    (lambda (xs ys)
      ;; theta: weights or input to target
      (lambda (theta)
	    (let ((pred-ys ((target xs) theta)))
	      (sum
	       (sqr
	        (- ys pred-ys))))))))

#|
Tests:
(define theta (list 0.0 0.0))

(numerize (((l2-loss line) line-xs line-ys) theta))
;;Value: 33.21

(define theta (list 0.0099 0.0))

(numerize (((l2-loss line) line-xs line-ys) theta))

;;Value: 32.59

(define theta (list 0.6263 0.0))

(numerize (((l2-loss line) line-xs line-ys) theta))

;;Value: 5.52

|#

#|
The Law of Revision

new theta = theta - (learning-rate * rate of change of loss with respect to theta)

Set learning rate to 0.01 so new theta is 0.0 - (0.01*-62.63) = 0.6263
|#

#|
Toy objective function
obj is the objective function that awaits a theta value because data (xs ys) has been provided already

In this case the objective is the l2 loss for a line given some set of features and outputs
|#
(define obj
  ((l2-loss line) line-xs line-ys))

#|
Tests:
(numerize (/ (- (obj (list (+ 0.6263 0.0099) 0.0))
		        (obj (list 0.6263 0.0)))
	         0.0099))
;Value: -25.12

;;; Getting various loss objective values

(numerize (obj (list -1.0 0.0))) ;Value: 126.21

(numerize (obj (list 0.0 0.0))) ;Value: 33.21

(numerize (obj (list 1.0 0.0))) ;Value: 0.21

(numerize (obj (list 2.0 0.0))) ;Value 27.21

(numerize (obj (list 3.0 0.0))) ;Value 114.21

(gradient-of (lambda (t) (sqr (ref t 0))) (list 27.0)) ;Value: (54.)

(gradient-of obj (list 0.0 0.0)) ;Value (-63. -21.)

|#


#|
Iteration procedure that will revise theta multiple times
---------------------------------------------------------
f: revision function which updates or revises theta
revs: number of revisions that theta will be subject to
t: theta (weights) which will be revised by f
|#

(define revise
  (lambda (f revs t)
    (cond
     ((zero? revs) t)
     (else
      (revise f (sub1 revs) (f t))))))

#|
Tests:
;; p. 81 Little Learner
(let ((f (lambda (t)
           (map (lambda (p)
	              (- p 3))
	            t))))
  (numerize (revise f 5 (list 1 2 3))))
;Value: (-14 -13 -12)

;; p. 84 Little Learner
(define learning-rate 0.01)

(define theta (list 0.0 0.0))

(let ((f (lambda (theta)
           (let ((gs (gradient-of obj theta)))
             (list
              (- (ref theta 0) (* learning-rate (ref gs 0)))
              (- (ref theta 1) (* learning-rate (ref gs 1))))))))
  (numerize (revise f 1000 theta)))
;Value: (1.05 1.87e-6)

(define revs 1000)

(define learning-rate 0.01)

(define theta (list 0.0 0.0))

(let ((f (lambda (theta)
           (let ((gs (gradient-of obj theta)))
             (map (lambda (p g)
	                (- p (* learning-rate g)))
	              theta
	              gs)))))
  (numerize (revise f revs theta)))
;Value: (1.05 1.87e-6)

|#


(define gradient-descent
  ;; obj: objective function
  ;; theta: weights or value being update to improve objective
  (lambda (obj theta)
    ;; f: revision function (gradient descent update to big-theta)
    (let ((f (lambda (big-theta)
               ;; p: element of big-theta
               ;; g: element of gradient with respect to big-theta
	           (map (lambda (p g)
		              (- p (* learning-rate g)))
		            big-theta
		            (gradient-of obj big-theta)))))
      (revise f revs theta))))

#|
Tests:
;; p. 90 Little Learner
(define theta (list 0.0 0.0))

(define revs 1000)

(define learning-rate 0.01)

(numerize (gradient-descent ((l2-loss line) line-xs line-ys)
                            theta))
;Value: (1.05 1.87e-6)

|#


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;;Quadratic
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;; toy quadratic dataset
(define quad-xs
  (tensor -1.0 0.0 1.0 2.0 3.0))

;; toy quadratic dataset
(define quad-ys
  (tensor 2.55 2.1 4.35 10.2 18.25))

;; quadratic function
;; y = ax^2 + bx + c
;; theta contains (a b c)
(define quad
  (lambda (t)
    (lambda (theta)
      (+ (* (ref theta 0) (sqr t))
	 (+ (* (ref theta 1) t) (ref theta 2))))))

#| Tests:
(numerize ((quad 3.0) (list 4.5 2.1 7.8))) ;Value: 54.6

|#

;; quadratic objective function assuming quadratic predictions
(define obj-quad
  ((l2-loss quad) quad-xs quad-ys))

#|
Tests:
(define learning-rate 0.0)

(define revs 1000)

(define learning-rate 0.001)

(numerize (gradient-descent obj-quad
                              (list 0.0 0.0 0.0)))
;Value: (1.48 0.99 2.05)

|#

#|
Current loss for the new theta values calculated from above objective
(define learning-rate 0.001)

(numerize (obj-quad (list 1.48 0.99 2.05)))
;Value 0.102

|#

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;;Plane
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

#|plane-xs and plane-ys have two different shapes and this is allowed as long as they have the same number of nested tensors
Both having 6 in this case
Easier to think of it is that both tensors need to have the same length
|#

(define plane-xs
  (tensor
   (tensor 1.0 2.05)
   (tensor 1.0 3.0)
   (tensor 2.0 2.0)
   (tensor 2.0 3.91)
   (tensor 3.0 6.13)
   (tensor 4.0 8.09)))

(define plane-ys
  (tensor 13.99
	  15.99
	  18.0
	  22.4
	  30.2
	  37.94))

;; target function (how predictions are made from inputs)
(define plane
  (lambda (t)
    (lambda (theta)
      (+ (dot-product (ref theta 0) t) (ref theta 1)))))

(define obj-plane
  ((l2-loss plane) plane-xs plane-ys))

#|
Determining the shape for theta is a little more complicated now
first is that theta is a list of parameters that are tensors, which can be different shapes
theta-1 must be the same shape  as the result of plane in this case is just a scalar
theta-0 needs to be the shape of a tensor-1 in plane-xs which in this case is (list 2)
|#

#|
Tests:
(define learning-rate 0.0)

(define revs 1000)

(define learning-rate 0.001)

(numerize (gradient-descent
	       obj-plane
	       (list (tensor 0.0 0.0) 0.0))) ;Value: ((tensor 3.98 2.05) 5.79)

(numerize
 ((plane (tensor 2.0 3.91))
  (list (tensor 3.98 2.04) 5.78))
 )                                      ;Value: 21.71

(numerize
 (+ (dot-product (tensor 3.98 2.04) (tensor 2.0 3.91))
    5.78)
 )                                      ;Value: 21.71

|#

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;;Sampling
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

#|
If time implemente batching for performance
|#
