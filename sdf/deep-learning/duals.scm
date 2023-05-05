;;; ==================================================
;;; Duals and Gradients using vectors/lists as tensors
;;; ==================================================

;; Dual Constructor
;; Duals are a way linking a number to the operations
;; that created the number (used in automatic differentiation)
;; -------------------------------------------------
;; real-component: a numerical value
;; link-component: a function managing the chain that produced
;;                 this scalar. Invoked for walking the chain
;;                 during automatic differentiation. A link
;;                 procedure takes in three arguments (a scalar
;;                 object, a multiplicative accumulator for chain
;;                 rule, and a gradient state table) and outputs
;;                 a mutated gradient state table. Note that the
;;                 scalar object inputted into the link procedure
;;                 should always be the dual/scalar itself whose
;;                 link is being called.
(define dual
  ;; creates vector of two components with
  ;; tag at the begining containing reference to
  ;; the constructor function
  (lambda (real-component link-component)
    (guarantee number? real-component dual)
    (guarantee procedure? link-component dual)
    (vector dual real-component link-component)))

;; Predicate testing whether an object is a dual
;; Returns #t iff object is a dual
(define (dual? object)
  (cond
   ((vector? object) (eq? (vector-ref object 0) dual))
  (else #f)))

;; Predicate testing whether an object is a scalar
;; Returns #t iff object is a number or a dual
(define (scalar? object)
  (cond
   ((number? object) #t)
   (else (dual? object))))

;; Gets the real component of a scalar object. Throws
;; an error when the object is not a scalar object
(define (real-component object)
  (guarantee scalar? object real-component)
  (cond
   ((dual? object) (vector-ref object 1))
   (else object)))

;; Gets the link component of a scalar object. A number
;; object has the end-of-chain function as its link component.
;; Throws an error when the object is not a scalar object.
(define (link-component object)
  (guarantee scalar? object link-component)
  (cond
   ((dual? object) (vector-ref object 2))
   (else end-of-chain)))

;; A link procedure which updates, by modifying the state-table of
;; some differentiable, the gradient with respect to scalar-self. The
;; update is performed by adding the accumulator (which is the
;; multiplicatively accumulated gradient with respect to scalar-self)
;; to the value associated with the scalar-self in the
;; state-table. Note that this summation is needed in cases like x + x
;; where the gradient with respect to x should be 2x but since the
;; values are split we find that the gradient is 1 + 1.  This
;; procedure is invoked on scalars that were not produced by anything
;; (in other words, the inputs)
;; ---------------------------------------------------------------------
;; scalar-self: the scalar leaf of a differentiable object, it
;; represents the current scalar node accumulator: a multiplicative
;; accumulator for the gradient components during chain rule
;; state-table: a state table for some differentiable. The table will
;; be mutated and returned afterwards
(define (end-of-chain scalar-self accumulator state-table)
  (check-link-input-types scalar-self accumulator state-table end-of-chain)
  (let ((scalar-lookup (hash-table-ref/default state-table scalar-self 0.0)))
    ;; NOTE1: the book uses hash-set in Racket which functionally modifies a
    ;; hash-table instead of mutating the original. This does not exist in MIT-Scheme
    ;; but it seems that for this scenario, ordinary mutation of the hash table is equivalent
    ;; given that the original, unmodified hash table is never used in the Racket version
    ;; NOTE2: the + operator here is the extended addition operater acting on tensors
    (hash-table-set! state-table scalar-self (+ accumulator scalar-lookup))
    state-table))

