;;;=============================================
;;; Tensors
;;;=============================================

;; operation to create tensors
(define tensor vector)

;; predicate for tensors
(define (tensor? object)
  (cond
   ((scalar? object) #t)
   ((vector? object) #t)
   (else
    #f)))

;; operation for referencing element of tensor by index
;; in the highest layer
;; ----------------------------------------------------
;; tensor-object: tensor of rank > 0
;; index: scalar >= 0   &   scalar < (tlen tensor-object)
(define (tref tensor-object index)
  (guarantee tensor? tensor-object tref)
  (guarantee scalar? index tref)
  (vector-ref tensor-object (real-component index)))

;; operation for determining length of tensor's highest layer
(define (tlen tensor-object)
  (guarantee tensor? tensor-object tlen)
  (vector-length tensor-object))

;; operation for referencing element of list by scalar index
;; ---------------------------------------------------------
;; list-object: a list
;; scalar-object: scalar >= 0   & scalar < (length list-object)
(define (ref list-object scalar-object)
  (guarantee list? list-object ref)
  (guarantee scalar? scalar-object ref)
  (list-ref list-object (real-component scalar-object)))

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
   (else (error t
                "object passed into ranked is invalid"))))

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
   (else (error t
                "object passed into shape is invalid"))))

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
    (error t
           "object t passed into of-rank? is invalid"))))

#|
Tests:
(of-rank? 3 #(#(#(8) #(9)) #(#(4) #(7))))
;Value: #t

(of-rank? 2 #(#(#(8) #(9)) #(#(4) #(7))))
;Value: #f

(of-rank? 4 #(#(#(8) #(9)) #(#(4) #(7))))
;Value: #f

|#


;; map operation for tensors
;; applies the function f to every element in tensor
;; object in the highest layer
;; -------------------------------------------------
;; f: a function
;; object: a tensor
(define (tmap f object)
  (guarantee tensor? object tmap)
  (vector-map f object))

#|
Tests:
(tmap add1 (tensor 3 5 4))
;Value: #(4 6 5)

(tmap sub1 #(10 9 8))
;Value: #(9 8 7)

|#
