(load "~/6.5151/6.5151-Final-Project/automatic-differentiation/0-duals.scm")

(define ref
  (lambda (lst i)
    (list-ref lst (real-component i))))

(define refr
  (lambda (lst i)
    (drop lst (real-component i))))

#|
Tensor Representation
|#

(define tref
  (lambda (t i)
    (vector-ref t (real-component i))))

;;;tensor will just create a vector

(define tensor vector)

(define tlen vector-length)

(define tmap vector-map)

(define list->tensor list->vector)


(define tensor?
  (lambda (t)
    (cond
     ((scalar? t) #t)
     ((vector? t) #t)
     (else #f))))

(define build-tensor
 (lambda (s f)
   (built-tensor f s '())))

(define built-tensor
  (lambda (f s idx)
    (cond
     ((= (length s) 1)
      (make-vector (ref s 0)
		   (lambda (i)
		     (write i)
		      (f (append idx (list i))))))
     (else
      (make-vector (ref s 0)
		   (lambda (i)
		     (write i)
		      (built-tensor f (refr s 1) (append idx (list i)))))))))

(define trefs
  (lambda (t b)
    (build-vector (length b)
		  (lambda (i)
		    (tref t (list-ref b i))))))

#|
Tensor Operations
|#


#|
Shape of the tensor is a n size list
Where the ith value is the size of the ith dimension of the tensor
Scalar shape is empty list
|#
(define shape
  (lambda (t)
    (cond
     ((scalar? t) '())
     (else (cons (tlen t) (shape (tref t 0)))))))

(define rank
  (lambda (t)
    (length (shape t))))

(define size-of
  (lambda (s)
    (sized s 1)))

(define sized
  (lambda (s a)
    (cond
     ((null? s) a)
     (else (sized (refr s 1) (* a (ref s 0)))))))

#|Reshaping a Tensor|#

(define reshape
  (lambda (s t)
    (cond
     ((= (size-of s) (size-of (shape t)))
      (reshape-tensor s t))
     (else (error "cannot reshape"
		  (shape t) s)))))

(define reshape-tensor
  (lambda (s t)
    (let ((t-strides (strides (shape t)))
	  (s-strides (strides s)))
      (build-tensor s
		    (lambda (idx)
		      (let ((t-idx
			     (convert-idx t-strides s-strides idx)))
			(deep-tref t t-idx)))))))

(define convert-idx
  (lambda (t-strides s-strides idx)
    (invert-reference t-strides
		      (flat-ref s-strides idx))))

(define strides
  (lambda (s)
    (cond
     ((null? s) '())
     (else (cons (size-of (refr s 1))
		 (strides (refr s 1)))))))

(define flat-reference
  (lambda (strides idx a)
    (cond
     ((null? strides) a)
     (else (flat-reference (refr strides 1) (refr idx 1)
			   (+ a (* (ref strides 0) (ref idx 0))))))))


(define invert-reference
  (lambda (stride idx)
    (cond
     ((null? stride) '())
     (else
      (cons (quotient idx (ref stride 0))
	    (invert-reference (refr stride 1)
			      (remainder idx (ref stride 0))))))))

(define deep-tref
  (lambda (t idx)
    (cond
     ((null? idx) t)
     (else
      (deep-tref (tref t (ref idx 0))
		 (refr idx 1))))))

