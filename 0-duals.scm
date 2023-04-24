;; Dual Constructor
;; Duals are a way linking a number to the operations
;; that created the number, used in automatic differentiation
;; -------------------------------------------------
;; real-component: a numerical value
;; link-component: a function managing the chain that produced
;;                 this scalar. Invoked for walking the chain
;;                 during automatic differentiation
(define dual
  ;; creates vector of two components with
  ;; tag at the begining containing reference to
  ;; the constructor
  (lambda (real-component link-component)
    (vector dual real-component link-component)))

;; Predicate testing whether an object is a dual
;; Returns #t iff object is a dual
(define (dual? object)
  (cond
   ((vector? object) (eq? (vector-ref object 0) dual))
  (else #f)))

;; Predicate testing whether an object is a scalar
;; Returns #t iff object is a real number or a dual
(define (scalar? object)
  (cond
   ((number? object) #t)
   (else (dual? object))))

;; Gets the real compnent of a scalar object. Throws
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

#|
Terminology from book:
differentiable functions := functions producing scalar loss
                            (not fully generalized)
differentiables := scalars, lists of differentiables, or
                   vectors of differentiables (recursive definition)
                   Note: differentiables can be thought of as DAGs
                   with scalars as leaves and lists/vectors as nodes
|#

;; Produces a new differentiable object contains scalar leaves such
;; that the function f is applied to the scalar leaves of the input
;; differentiable object
;; ------------------------------------------------------------------
;; f: a function taking in a single scalar as input
;; object: a differentiable object (see definition above)
(define (differentiable-map f object)
  (cond
   ((scalar? object) (f object))
   ((list? object) (map (lambda (list-member)
                          (differentiable-map f list-member))
                        object))
   ((vector? object) (map (lambda (vector-element)
                            (differentiable-map f vector-element))
                          object))
   (else (error object
                "object passed into differentiable-map is not a differentiable object"))))

;; Converts a scalar object to a truncated dual which is a dual object
;; whose link-component is the end-of-chain function.
(define (truncated-dual object)
  (dual (real-component object)
        end-of-chain))
