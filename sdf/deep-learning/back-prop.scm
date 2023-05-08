

(define line-xs
  (tensor 2.0 1.0 4.0 3.0))

(define line-ys
  (tensor 1.8 1.2 4.2 3.3))

(define theta
  (list 0.0 0.0))

(numerize ((line line-xs) theta))

#|
Defines loss for a given dataset, however with sum, negative numbers can lead to a loss of 0 for something that is not an ideal fit
|#

(numerize (sum
	   (- line-ys ((line line-xs) theta)))
	  )
#|
TO fix the issue with negative values, we square each element first before taking the sum
|#

(numerize
 (sum
  (sqr
   (- line-ys ((line line-xs) theta))))
 )

#|
(define l2-loss
  (lambda (xs ys)
    (lambda (t)
      (let ((pred-ys ((line xs) t)))
	(sum
	 (sqr
	  (- ys pred-ys)))))))

(numerize ((l2-loss line-xs line-ys) theta))
					;Value: 33.21
|#

(define l2-loss
  (lambda (target)
    (lambda (xs ys)
      (lambda (theta)
	(let ((pred-ys ((target xs) theta)))
	  (sum
	   (sqr
	    (- ys pred-ys))))))))

(numerize (((l2-loss line) line-xs line-ys) theta)) ;;Value: 33.21

(define theta
  (list 0.0099 0.0))

(numerize (((l2-loss line) line-xs line-ys) theta)) ;;Value: 32.59

#|
change in loss was roughly -0.62 so the rate of change is -0.62/0.0099 = -62.63. So for small change in theta we get much larger change in loss. 
|#

#|
The Law of Revision

new theta = theta - (learning-rate * rate of change of loss with respect to theta)

Set learning rate to 0.01 so new theta is 0.0 - (0.01*-62.63) = 0.6263
|#

(define learning-rate
  0.01)

(define theta
  (list 0.6263 0.0))

(numerize (((l2-loss line) line-xs line-ys) theta)) ;;Value: 5.52

#|
Procedures to get rate of change of loss for theta at 0.6263

obj is the objective function that awaits a theta value
|#

(define obj
  ((l2-loss line) line-xs line-ys))

(numerize (/ (- (obj (list (+ 0.6263 0.0099) 0.0))
		(obj (list 0.6263 0.0)))
	     0.0099)
	  )
;;Value: -25.12

#|
Getting various loss values
|#

(numerize (obj (list -1.0 0.0))) ;Value: 126.21
(numerize (obj (list 0.0 0.0))) ;Value: 33.21
(numerize (obj (list 1.0 0.0))) ;Value: 0.21
(numerize (obj (list 2.0 0.0))) ;Value 27.21
(numerize (obj (list 3.0 0.0))) ;Value 114.21

(gradient-of (lambda (t) (sqr (ref t 0))) (list 27.0)) ;Value: (54.)

(gradient-of obj (list 0.0 0.0)) ;Value (-63. -21.)

#|
Iteration procedure that will revise theta multiple times
|#

(define revise
  (lambda (f revs t)
    (cond
     ((zero? revs) t)
     (else
      (revise f (sub1 revs) (f t))))))

(define f-map
  (lambda (t)
    (map (lambda (p)
	   (- p 3))
	 t)))

(numerize (revise f-map 5 (list 1 2 3))) ;Value: (-14 -13 -12)
	  
(define learning-rate
  0.01)
#|
f is function to be applied to theta that will adjust the values to reduce loss
|#

(define f
  (lambda (theta)
    (let ((gs (gradient-of obj theta)))
      (list
       (- (ref theta 0) (* learning-rate (ref gs 0)))
       (- (ref theta 1) (* learning-rate (ref gs 1)))))))

(numerize (revise f 1000 (list 0.0 0.0))) ;Value: (1.05 1.87e-6)

(define revs 1000)

(define f
  (lambda (theta)
    (let ((gs (gradient-of obj theta)))
      (map (lambda (p g)
	     (- p (* learning-rate g)))
	   theta
	   gs))))

(numerize (revise f revs (list 0.0 0.0))
	  ) ;Value: (1.05 1.87e-6)

(define gradient-descent
  (lambda (obj theta)
    (let ((f (lambda (th)
	       (map (lambda (p g)
		      (- p (* learning-rate g)))
		    th
		    (gradient-of obj th)))))
      (revise f revs theta))))

(numerize (gradient-descent
 ((l2-loss line) line-xs line-ys)
 (list 0.0 0.0))
	  ) ;Value: (1.05 1.87e-6)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;;Quadratic
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


#|
The graph from quad dataset does not form a line unlike the one from the line dataset
|#
(define quad-xs
  (tensor -1.0 0.0 1.0 2.0 3.0))

(define quad-ys
  (tensor 2.55 2.1 4.35 10.2 18.25))

(define quad
  (lambda (t)
    (lambda (theta)
      (+ (* (ref theta 0) (sqr t))
	 (+ (* (ref theta 1) t) (ref theta 2))))))

(numerize ((quad 3.0) (list 4.5 2.1 7.8))) ;Value: 54.6

(define obj-quad
  ((l2-loss quad) quad-xs quad-ys))

(define learning-rate 0.001)

(numerize (gradient-descent
	   obj-quad
	   (list 0.0 0.0 0.0))) ;Value: (1.48 0.99 2.05)

#|Current loss for the new theta values calculated from above gradient descent|#

(numerize (obj-quad (list 1.48 0.99 2.05))) ;Value 0.102

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

(numerize (gradient-descent
	   obj-plane
	   (list (tensor 0.0 0.0) 0.0))) ;Value: ((tensor 3.98 2,05) 5.79)

(numerize
 ((plane (tensor 2.0 3.91))
  (list (tensor 3.98 2.04) 5.78))
 ) ;Value: 21.71

(numerize
 (+ (dot-product (tensor 3.98 2.04) (tensor 2.0 3.91))
    5.78)
 ) ;Value: 21.71

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;;Sampling
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

#|
If time implemente batching for performance
|#
