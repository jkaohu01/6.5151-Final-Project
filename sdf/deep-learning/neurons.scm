(manage 'new 'deep-learning)

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



