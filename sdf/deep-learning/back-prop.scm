(manage 'new 'deep-learning)

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
      (lambda (t)
	(let ((pred-ys ((target xs) t)))
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
  (lambda (t)
    (let ((gs (gradient-of obj t)))
      (list
       (- (ref t 0) (* learning-rate (ref gs 0)))
       (- (ref t 1) (* learning-rate (ref gs 1)))))))

(numerize (revise f 1000 (list 0.0 0.0))) ;Value: (1.05 1.87e-6)

(define revs 1000)

(define f
  (lambda (t)
    (let ((gs (gradient-of obj t)))
      (map (lambda (p g)
	     (- p (* learning-rate g)))
	   t
	   gs))))

(numerize (revise f revs (list 0.0 0.0))
	  ) ;Value: (1.05 1.87e-6)

(define gradient-descent
  (lambda (obj t)
    (let ((f (lambda (th)
	       (map (lambda (p g)
		      (- p (* learning-rate g)))
		    th
		    (gradient-of obj th)))))
      (revise f revs t))))

(numerize (gradient-descent
 ((l2-loss line) line-xs line-ys)
 (list 0.0 0.0))
	  ) ;Value: (1.05 1.87e-6)
