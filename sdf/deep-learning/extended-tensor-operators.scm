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
((line 7.3) (list 1.0 0))
;Value: 7.3

((line 2) (list 4 -1))
Value: 7

|#

;; debugging function for converting the scalar leaves of some
;; tensor differentiable from duals to numbers. This makes it
;; easier to read when printed as output because the duals are
;; not shown in their object form with links procedures
;; -----------------------------------------------------------
;; t: a tensor (all tensors are differentiables)
(define (numerize-tensor t)
  (guarantee tensor? t numberify-tensor)
  (differentiable-map real-component t))

;; alias for numerize-tensor
(define numerize numerize-tensor)

;; Procedure for extending single input functions to tensors.
;; The input function f will be applied to the the nested
;; tensors of rank base-rank to produce a new tensor.
;; ----------------------------------------------------------
;; f: some function operating on one tensor argument of rank base-rank
;; base-rank: the numeric rank of the tensors for which f operates on
(define (ext1 f base-rank)
  (guarantee procedure? f ext1)
  (guarantee number? base-rank ext1)
  ;; tensor-object: some tensor
  (lambda (tensor-object)
    (cond
     ;; apply the operation to the tensors with correct rank
     ((of-rank? base-rank tensor-object) (f tensor-object))
     ;; descend into nested tensors to apply f
     ((tensor? tensor-object) (tmap (ext1 f base-rank) tensor-object))
     (else (error tensor-object
                  "the tensor-object passed into lambda output of ext1 is invalid")))))

;; Produces a function taking a single tensor as input and outputs a
;; tensor whose scalars have the Scheme sqrt function applied to them
;; Note: a tensor with duals as scalars is produced as output
;; -----------------------------------------------------------------------
(define sqrt
  (ext1 sqrt-0 0))

#|
Tests:
(numerize (sqrt #(#(1 4) #(9 16))))
;Value: #(#(1 2) #(3 4))

|#

;; Produces a function taking a single tensor as input and outputs a
;; tensor whose scalars are converted to 0.0
(define zeroes
  (ext1 (lambda (x) 0.0) 0))

#|
Tests:
(zeroes #(#(#(1 2 3) #(4 5 6)) #(#(7 8 9) #(10 11 12))))
;Value: #(#(#(0. 0. 0.) #(0. 0. 0.)) #(#(0. 0. 0.) #(0. 0. 0.)))

|#

;; Function for finding the sum of elements of rank 1 tensor
;; ----------------------------------------------------------
;; t: a rank 1 tensor
(define (sum-1 t)
  (guarantee (lambda (x) (of-rank? 1 x)) t sum-1)
  (summed t (sub1 (tlen t)) 0))

;; helper for 1:sum
(define (summed t index accumulator)
  (cond
   ((zero? index) (+ (tref t 0) accumulator))
   (else
    (summed t (sub1 index) (+ (tref t index) accumulator)))))

#|
Tests:
(sum-1 #(10.0 12.0 14.0))
;Value: 36.

(sum-1 #(1 2 3))
;Value: 6

|#

;; Produces a function taking a single tensor as input and outputs a
;; tensor whose rank 1 tensors are folded additively into a scalar
;; Note: a tensor with duals as scalars is produced as output
(define sum
  (ext1 sum-1 1))

#|
Tests:
(sum #(#(#(1 2) #(3 4)) #(#(5 6) #(7 8))))
;Value: #(#(3 7) #(11 15))

|#

;; Takes a rank 2 tensor and flattens it into a rank 1 tensor. This means
;; that the elements of the rank 2 tensor are concatenated, in the order
;; they appear, into a single rank 1 tensor
;; -----------------------------------------------------------------------
;; t: a tensor of rank 2
(define (flatten-2 t)
  (guarantee (lambda (x) (of-rank? 2 x)) t flatten-2)
  (let ((s (shape t)))
    (let ((rows (ref s 0))
          (cols (ref s 1)))
      (make-initialized-vector (s:* rows cols)
                               (lambda (initializer-index)
                                 (tref
                                  (tref t (quotient initializer-index cols))
                                  (remainder initializer-index cols)))))))

#|
Tests:
(flatten-2 #(#(1.0 0.5) #(3.1 2.2) #(7.3 2.1)))
;Value: #(1. .5 3.1 2.2 7.3 2.1)

|#


;; Produces a function taking a single tensor as input and outputs a
;; tensor whose innermost rank 2 tensors are flattened into a rank 1
;; tensors.
(define flatten
  (ext1 flatten-2 2))

#|
Tests:
(flatten #(#(#(1.0 0.5) #(3.1 2.2) #(7.3 2.1)) #(#(2.9 3.5) #(0.7 1.5) #(2.5 6.4))))
;Value: #(#(1. .5 3.1 2.2 7.3 2.1) #(2.9 3.5 .7 1.5 2.5 6.4))

|#

