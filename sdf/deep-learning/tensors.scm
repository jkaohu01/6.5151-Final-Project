;;;=============================================
;;; Tensors
;;;=============================================

;; operation to create tensors
;; alias to vector
(define tensor vector)

#|
Tests:
(tensor 1 2 3)
;Value: #(1 2 3)

(tensor (tensor 1) (tensor 2))
;Value: #(#(1) #(2))

|#

;; predicate for tensors
;; -----------------------
;; object: any object
(define (tensor? object)
  (cond
   ((scalar? object) #t)
   ((vector? object) #t)
   (else
    #f)))

#|
Tests:
(tensor? 1)
;Value: #t

(tensor? (+ 1 1))
;Value: #t

(tensor? #(1 2))
;Value: #t

(tensor? 'tensor)
;Value: #f

|#

;; predicate for determinine whether this is a tensor above rank 0
;; ---------------------------------------------------------------
;; t: any object
(define (tensor-above-rank-0? t)
  (and (tensor? t) (not (of-rank? 0 t))))

#|
Tests:
(tensor-above-rank-0? 2)
;Value: #f

(tensor-above-rank-0? #(1))
;Value: #t

|#

;; predicate for testing whether a scalar value is within the bounds
;; 0 <= value < upper. #t returned iff this statement is true
;; ----------------------------------------------------------------
;; upper: some upper bound
;; value: some scalar value
(define (scalar-within-len-bounds? upper)
  (guarantee number? upper scalar-within-len-bounds?)
  (named-lambda (scalar-within-len-bounds? value)
    (guarantee scalar? value scalar-within-len-bounds?)
    (and (s:<= 0 (real-component value)) (s:< (real-component value) upper))))

#|
Tests:
((scalar-within-len-bounds? 1) 2)
;Value: #f

((scalar-within-len-bounds? 1) 0)
;Value: #t

((scalar-within-len-bounds? 1) 1)
;Value: #f

((scalar-within-len-bounds? 1) -1)
;Value: #f

|#

;; predicate for testing whether a scalar value is within the bounds
;; lower <= value <= upper. #t returned iff this statement is true
;; ----------------------------------------------------------------
;; upper: some upper bound
;; value: some scalar value
(define (scalar-within-closed-interval? lower upper)
  (guarantee number? upper scalar-within-closed-interval?)
  (named-lambda (scalar-within-closed-interval? value)
    (guarantee scalar? value scalar-within-closed-interval?)
    (and (s:<= lower (real-component value)) (s:<= (real-component value) upper))))

#|
Tests:
((scalar-within-closed-interval? -1 1) -1)
;Value: #t

((scalar-within-closed-interval? -1 1) 0)
;Value: #t

((scalar-within-closed-interval? -1 1) 1)
;Value: #t

((scalar-within-closed-interval? -1 1) -2)
;Value: #f

((scalar-within-closed-interval? -1 1) 2)
;Value: #f

|#

;; operation for determining length of tensor's highest layer
;; ----------------------------------------------------------
;; t: a tensor with rank > 0
(define (tlen t)
  (guarantee tensor-above-rank-0? t tlen)
  (vector-length t))

#|
Tests:
(tlen #(0 1 2))
;Value: 3

(tlen #(#(0 1) #(2 3)))
;Value: 2

|#

;; operation for referencing element of tensor by index
;; in the highest layer
;; ----------------------------------------------------
;; t: tensor of rank > 0
;; index: a scalar where 0 <= index < (tlen t)
(define (tref t index)
  (guarantee tensor-above-rank-0? t tref)
  (guarantee scalar? index tref)
  (guarantee (scalar-within-len-bounds? (tlen t)) index tref)
  (vector-ref t (real-component index)))

#|
Tests:
(tref (tensor 0 1 2) 0)
;Value: 0

(tref (tensor 0 1 2) 2)
;Value: 2

(tref #(#(1 2) #(3 4)) 1)
;Value: #(3 4)

|#

;; operation for creating a new tensor from an old tensor and
;; a list of references indicies to the old tensor. The new
;; tensor will contain only entries from the old tensor. That is,
;; the ith entry of the new tensor will be (tref t j-index) where
;; j-index is the ith entry of indices
;; -------------------------------------------------------------
;; t: tensor with rank > 0
;; indices: a list of reference indices to t
(define (trefs t indices)
  (guarantee tensor-above-rank-0? t trefs)
  (guarantee list? indices trefs)
  (let ((indices% (list->vector indices)))
    (make-initialized-vector (vector-length indices%)
                             (lambda (init-vec-index)
                               (tref t (vector-ref indices% init-vec-index))))))

#|
Tests:
(trefs #(-0 -1 -2 -3 -4 -5 -6) (list 6 0 3 1))
;Value: #(-6 0 -3 -1)

(trefs #(#(0 0) #(1 -1) #(2 -2) (3 -3) (4 -4)) (list 2 0 4))
;Value: #(#(2 -2) #(0 0) (4 -4))

|#

;; operation for finding size of list
;; -----------------------------------------------------
;; lst : a list object
(define (len lst)
  (guarantee list? lst len)
  (length lst))

#|
Tests:
(len '(0 1))
;Value: 2

(len '(0))
;Value: 1

(len '())
;Value: 0

|#

;; operation for referencing element of list by scalar index
;; ---------------------------------------------------------
;; list-object: a list
;; index: a scalar where 0 <= index < (tlen tensor-object)
(define (ref list-object index)
  (guarantee list? list-object ref)
  (guarantee scalar? index ref)
  (guarantee (scalar-within-len-bounds? (length list-object)) index ref)
  (list-ref list-object (real-component index)))

#|
Tests:
(ref '(0 1 2) 2)
;Value: 2

(ref '(0 1 2) 0)
;Value: 0

|#

;; operation for dropping first i elements of list
;; -----------------------------------------------
;; list-object: a list
;; i: the number of elements that should be dropped from list head
;;    where 0 <= i <= (length list-object)
(define (refr list-object i)
  (guarantee list? list-object refr)
  (guarantee scalar? i ref)
  (guarantee (scalar-within-closed-interval? 0 (length list-object)) i refr)
  (drop list-object (real-component i)))

#|
Tests:
(refr '(0 1 2) 0)
;Value: (0 1 2)

(refr '(0 1 2) 1)
;Value: (1 2)

(refr '(0 1 2) 2)
;Value: (2)

(refr '(0 1 2) 3)
;Value: ()

|#

;; Function for finding the rank of a tensor (the depth
;; of the nesting)
;; ----------------------------------------------------
;; t: a tensor
(define (rank t)
  (guarantee tensor? t rank)
  (ranked t 0))

;; Helper for rank which accumulates rank
(define (ranked t accumulator)
  (cond
   ((scalar? t) accumulator)
   ((tensor? t) (ranked (tref t 0) (add1 accumulator)))
   (else
    (bad-arg-error 't 'ranked))))

#|
Tests:
(rank (tensor (tensor (tensor 8) (tensor 9)) (tensor (tensor 4) (tensor 7)))) 
;Value: 3

(rank #(#(#(8) #(9)) #(#(4) #(7)))) 
;Value: 3

(rank #(#(5.2 6.3 8.0) #(6.9 7.1 0.5))) 
;Value: 2

(rank #(#(#(5) #(6) #(8)) #(#(7) #(9) #(5)))) 
;Value: 3

(rank #(9 4 7 8 0 1)) 
;Value: 1

(rank 3) 
;Value: 0

|#

;; Function for determining the shape of a tensor
;; ----------------------------------------------------
;; t: a tensor
(define (shape t)
  (cond
   ((scalar? t) (list))
   ((tensor? t) (cons (tlen t) (shape (tref t 0))))
   (else (bad-arg-error 't 'shape))))

#|
Tests:
(shape #(#(5.2 6.3 8.0) #(6.9 7.1 0.5)))
;Value: (2 3)

(shape #(#(#(5) #(6) #(8)) #(#(7) #(9) #(5))))
;Value: (2 3 1)

(shape #(9 4 7 8 0 1))
;Value: (6)

(shape 3)
;Value: ()

|#

;; predicate for testing whether a tensor's rank matches some value
;;-----------------------------------------------------------------
;; n: some number
;; t: some tensor
(define (of-rank? n t)
  (guarantee number? n of-rank?)
  (cond
   ((zero? n) (scalar? t))
   ((scalar? t) #f)
   ((tensor? t) (of-rank? (sub1 n) (tref t 0)))
   (else
    (bad-arg-error 't 'of-rank?))))

#|
Tests:
(of-rank? 3 #(#(#(8) #(9)) #(#(4) #(7))))
;Value: #t

(of-rank? 2 #(#(#(8) #(9)) #(#(4) #(7))))
;Value: #f

(of-rank? 4 #(#(#(8) #(9)) #(#(4) #(7))))
;Value: #f

|#

;; predicate for testing whether an object is a tensor and whether the
;; tensor's rank matches some value.  Useful for guarantee statements
;; where only one argument is accepted in a predicate.
;; -----------------------------------------------------------------------------
;; n: some number
(define (correct-rank-tensor? n)
  (guarantee number? n correct-rank-tensor?)
  ;; t: some tensor
  (named-lambda (correct-rank-tensor? t)
    (and (tensor? t) (of-rank? n t))))

;; compares the rank of two tensors. Returns #t iff (rank t) > (rank u)
;; --------------------------------------------------------------------
;; t: a tensor
;; u: a tensor
(define (rank> t u)
  (cond
   ((and (scalar? t) (tensor? u)) #f)
   ((and (scalar? u) (tensor? t)) #t)
   ((and (tensor? t) (tensor? u)) (rank> (tref t 0) (tref u 0)))
   (else
    (or-bad-args-error 't 'u 'rank>))))

#|
Tests:
(rank> #(1) 2)
;Value: #t

(rank> 2 #(1))
;Value: #f

(rank> #(#(1)) #(2))
;Value: #t

(rank> #(2) #(#(1)))
;Value: #f

(define t1-rank> #(#(#(1 2) #(3 4) #(5 6)) #(#(-1 -2) #(-3 -4) #(-5 -6))))
;Value: t1-rank>

(define u1-rank> #(#(11 2) #(3 1)))
;Value: u1-rank>

(rank> t1-rank> u1-rank>)
;Value: #t

(rank> u1-rank> t1-rank>)
;Value: #f

|#

;; predicate for ensuring that two tensors have a certain rank
;; Returns #t iff both tensors have their specified rank
;; -----------------------------------------------------------
;; n: a number indicating the rank that t should have
;; t: a tensor
;; m: a number indicating the rank that u should have
;; u: a tensor
(define (of-ranks? n t m u)
  (guarantee number? n of-ranks?)
  (guarantee tensor? t of-ranks?)
  (guarantee number? m of-ranks?)
  (guarantee tensor? u of-ranks?)
  (cond
   ((of-rank? n t) (of-rank? m u))
   (else
    #f)))

#|
Tests:
(of-ranks? 3 #(#(#(8) #(9)) #(#(2) #(1))) 2 #(5))
;Value: #f

(of-ranks? 3 #(#(#(8) #(9)) #(#(2) #(1))) 2 #(#(5)))
;Value: #t

|#

;; map operation for tensors
;; also an alias for vector-map with some helpful error messages
;; -------------------------------------------------------------
;; f: some function to be applied to the tensor-args to construct
;;    a new tensor
;; tensor-args: tensors that f will draw from 
(define (tmap f . tensor-args)
  (guarantee procedure? f tmap)
  ;; ensure all arguments are tensors
  ;; for fast error detection
  (for-each (lambda (tensor-arg)
              (guarantee tensor-above-rank-0? tensor-arg tmap))
            tensor-args)
  (apply vector-map f tensor-args))

#|
Tests:
(tmap add1 (tensor 3 5 4))
;Value: #(4 6 5)

(tmap sub1 #(10 9 8))
;Value: #(9 8 7)

(tmap (lambda (x y) (s:+ x y)) #(1 2 3) #(-1 -2 -3))
;Value: #(0 0 0)

|#
