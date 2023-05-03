;;; ===============================================
;;; Overriding of primitive operators to new symbol
;;; ===============================================
(define (base-scheme symbol)
  (environment-lookup system-global-environment symbol))
(define s:+ (base-scheme '+))
(define s:- (base-scheme '-))
(define s:* (base-scheme '*))
(define s:/ (base-scheme '/))
(define s:exp (base-scheme 'exp))
(define s:< (base-scheme '<))
(define s:> (base-scheme '>))
(define s:<= (base-scheme '<=))
(define s:>= (base-scheme '>=))
(define s:= (base-scheme '=))
(define s:expt (base-scheme 'expt))
(define s:log (base-scheme 'log))
(define s:sqrt (base-scheme 'sqrt))

;; Subtracts 1 from the numeric value
(define (sub1 numeric-value)
  (guarantee number? numeric-value sub1)
  (s:- numeric-value 1))

;; Adds 1 to the numeric value
(define (add1 numeric-value)
  (guarantee number? numeric-value add1)
  (s:+ numeric-value 1))

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

#|
Terminology from book:
differentiable functions (f) := functions producing scalar loss
                                (not fully generalized)

differentiables (θ) := scalars, lists of differentiables, or
                       vectors of differentiables (recursive definition)
                       Note: differentiables can be thought of as DAGs
                       with scalars as leaves and lists/vectors as nodes

truncated differentiable := a differentiable but with its leaves being
                            truncated dual objects.

truncated dual := like a dual but its link-component is the end-of-chain
                  function

|#

;; Quick test for whether an object is a differentiable. If #f is returned,
;; then the object is guaranteed to not be a differentiable, but if #t is
;; returned the object may be a differentiable but it also may not. A true
;; test would require a recursive search which may be computationally expensive
(define (maybe-differentiable? object)
  (or (scalar? object) (list? object) (vector? object)))

;; Produces a new differentiable object containing scalar leaves such
;; that the function f is applied to the scalar leaves of the input
;; differentiable object.
;; Throws an error when the object is not a differentiable.
;; ------------------------------------------------------------------
;; f: a function taking in a single scalar as input
;; object: a differentiable object
(define (differentiable-map f object)
  (guarantee procedure? f differentiable-map)
  (guarantee maybe-differentiable? object differentiable-map)
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

;; Converts a scalar object to a truncated dual
(define (truncated-dual object)
  (guarantee scalar? object truncated-dual)
  (dual (real-component object)
        end-of-chain))

;; Computes the gradient of a function f with respect to a tensor theta.
;; In the case that f outputs a single scalar, this computes the true
;; gradient. However, in the case that f outputs a tensor, this produces
;; something different. In this case, a theta-shaped gradient should be
;; produced for every scalar in the tensor output of f. But this function
;; will instead produce a single theta-shaped gradient that is the sum of
;; all the individual theta-shaped gradients. See "The Little Learner"
;; p. 362 frame 37 for reference.
;; --------------------------------------------------------------------
;; f: function for which the gradient is being sought. f must be a function
;;    that takes a theta-shaped tensor as input.
;; theta: the argument to f with respect to which we're seeking the gradient.
;;        theta must be a tensor (which guarantees it is a differentiable)
(define (gradient-of f theta)
  (guarantee procedure? f gradient)
  (guarantee maybe-differentiable? theta gradient)
  ;; The let statement below transforms the theta (differentiable) into a truncated
  ;; differentiable that abandons prior links that the scalars in theta may contain.
  ;; This allows us to ignore the history of theta and focus on what f performs
  ;; in this moment in time. See p. 365 frame 45 in "The Little Learner"
  (let ((with-respect-to (differentiable-map truncated-dual theta)))
    ;; we produce a differentiable by calling f on with-respect-to
    ;; and then find the gradient-once of this differentiable with respect
    ;; to with-respect-to. 
    (gradient-once (f with-respect-to) with-respect-to)))

#|
Terminology from book:
with-respect-to := a truncated differentiable that is passed as input to some function f

y := the scalar output from f(with-respect-to). Can also be a tensor in the general case
     but the gradient state will be something different.

gradient state (σ) := a mapping between scalars and gradients such that every scalar d in
                      with-respect-to is associated with an accumulator that represents the
                      current gradient of y with respect to d. Technically, the mapping will
                      include every scalar that produced y, even scalars not in with-respect-to,
                      but we will ignore these since they are constants

Note: in the case that y is a tensor, the gradient state is the sum of all the with-respect-to shaped
gradients of every scalar in y with respect to every scalar d from with-respect-to. As before,
technically it will include every scalar d that produced y.
|#

;; predicate for whether an object is a state-table (a gradient state)
(define (state-table? object)
  (hash-table? object))

;; Produces a differentiable whose shape is equivalent to the with-respect-to
;; differentiable except that every scalar leaf 'd' of with-respect-to is
;; replaced with the gradient of 'y' with respect to that scalar leaf 'd'
;; from with-respect-to
;; ----------------------------------------------------------------------------
;; y: a differentiable object produced as the output of some function f with
;;    the input with-respect-to (the following argument)
;; with-respect-to: a truncated differentiable object that is the input to some
;;                  function f that produced the differentiable object y
(define (gradient-once y with-respect-to)
  (guarantee maybe-differentiable? y gradient-once)
  (guarantee maybe-differentiable? with-respect-to gradient-once)
  ;; state-table will store the gradients of y with respect to each scalar
  ;; that produced y, including but not limited to, the scalars from with-respect-to
  (let ((state-table (gradient-state y (make-strong-eq-hash-table))))
    ;; create a new differentiable from with-respect-to where the scalar
    ;; leaves are replaced with with the gradient of y with respect to
    ;; that leaf
    (differentiable-map (lambda (scalar-leaf)
                          (guarantee scalar? scalar-leaf gradient-once)
                          (hash-table-ref/default state-table scalar-leaf 0.0))
                        with-respect-to)))

;; Mutates a gradient state table through the accumulation of the
;; gradients of a differentiable.  The input table will be updated
;; such that it contains the gradients of differentiable 'y' with
;; respect to every scalar in with-respect-to. This is accomplished by invoking
;; the link procedure for the every scalar in y (which results in walking up
;; the chain of procedures and scalars that produced y until we reach some input)
;; ----------------------------------------------------------------------------
;; y: a differentiable object
;; state-table: a gradient state accumulator table
(define (gradient-state y state-table)
  (guarantee maybe-differentiable? y gradient-state)
  (guarantee state-table? state-table gradient-state)
  (cond
   ((scalar? y)
    (let ((link-procedure (link-component y)))
      ;; base case: reached a scalar leaf of some differentiable
      ;; the link procedure for this scalar will be called with the initial multiplicative
      ;; accumulator of 1.0 (identity). The chain of operations that produced this scalar will
      ;; be walked backwards until we reach scalars with the end-of-chain function (input scalars).
      ;; The table will be updated to reflect the gradient influence that each input scalar had on
      ;; the output of this chain (which is the current y scalar)
      (link-procedure y 1.0 state-table)))
   ((list? y)
    ;; recursive case: recurse until we reach a scalar
    (gradient-state-list y state-table))
   ((vector? y)
    ;; recursive case: recurse until we reach a scalar 
    (gradient-state-vec y (sub1 (vector-length y)) state-table))
   (else (error object
                "object passed into gradient-state is not a differentiable object"))))

;; Helper for gradient-state that explores all the branches of the differentiable
;; list y in index order and updates the state-table appropriately
;; ------------------------------------------------------------------------------
;; y: a differentiable list object
;; state-table: a gradient state accumulator table
(define (gradient-state-list y state-table)
  (guarantee list? y gradient-state-list gradient-state-list)
  (guarantee state-table? state-table gradient-state-list)
  (cond
   ;; when no more branches to explore, return updated state table
   ((null? y) state-table)
   (else
    ;; first update the state table by exploring the first branch of y
    (let ((state-table-updated (gradient-state (car y) state-table)))
      ;; next update the state table by exploring the remaining branches of y
      (gradient-state-list (cdr y) state-table-updated)))))

;; Helper for gradient-state that explores all the branches of the differentiable
;; vector y in reverse index order and updates the state-table appropriately
;; ------------------------------------------------------------------------------
;; y: a differentiable vector object
;; explore-index: an index into the y vector that we should recursively explore
;; state-table: a gradient state accumulator table
(define (gradient-state-vec y explore-index state-table)
  (guarantee vector? y gradient-state-vec)
  (guarantee number? explore-index gradient-state-vec)
  (guarantee state-table? state-table gradient-state-vec)
  ;; first update the state table by exploring the explore-index branch of y
  (let ((state-table-updated (gradient-state (vector-ref y explore-index) state-table)))
    (cond
     ;; when we've explored all branches in reverse-branch exploration, return
     ;; updated state table
     ((zero? explore-index) state-table-updated)
     (else
      ;; next update the state table by exploring the next explore-index branch of y
      ;; in reverse branch order
      (gradient-state-vec y (sub1 explore-index) state-table-updated)))))

;;;========================================
;;; Links and Helpers for Creating Link
;;; procedures which assist in automatic
;;; differentiation
;;;========================================

;; runs tests on the inputs to ensure correct types for a link procedure
(define (check-link-input-types scalar-leaf accumulator state-table caller)
  (guarantee scalar? scalar-leaf caller)
  (guarantee number? accumulator caller)
  (guarantee state-table? state-table caller))

;; A link procedure which updates, by modifying the state-table of some differentiable, the gradient with
;; respect to scalar-self. The update is performed by adding the accumulator (which is the multiplicatively
;; accumulated gradient with respect to scalar-self) to the value associated with the scalar-self in the
;; state-table. Note that this summation is needed in cases like x + x where the gradient with respect
;; to x should be 2x but since the values are split we find that the gradient is 1 + 1. 
;; This procedure is invoked on scalars that were not produced by anything (in other words, the inputs)
;; -----------------------------------------------------------------------------------------------
;; scalar-self: the scalar leaf of a differentiable object, it represents the current scalar node
;; accumulator: a multiplicative accumulator for the gradient components during chain rule
;; state-table: a state table for some differentiable. The table will be mutated and returned afterwards
(define (end-of-chain scalar-self accumulator state-table)
  (check-link-input-types scalar-self accumulator state-table end-of-chain)
  (let ((scalar-lookup (hash-table-ref/default state-table scalar-self 0.0)))
    ;; NOTE: the book uses hash-set in Racket which functionally modifies a
    ;; hash-table instead of mutating the original. This does not exist in MIT-Scheme
    ;; but it seems that for this scenario, ordinary mutation of the hash table is equivalent
    ;; given that the original, unmodified hash table is never used in the Racket version
    (hash-table-set! state-table scalar-self (s:+ accumulator scalar-lookup))
    state-table))

;; A helper procedure that helps define primitives for use in automatic
;; differentiation. It returns a procedure which takes in a dual as an
;; argument and produces another dual as output. The dual produced has a
;; link procedure that has access to the parents of this dual and a way of
;; accumulating gradients during the chain rule
;; ----------------------------------------------------------
;; numeric-function: a function taking 1 number as input and
;;                   producing 1 number as output
;; gradient-function: a function used to accumulate gradient
;;                    results during chain rule. Describes how
;;                    the numeric-function should be combined
;;                    to produce the chain rule result. This
;;                    should produce 1 value
(define (prim1 numeric-function gradient-function)
  (guarantee procedure? numeric-function prim1)
  (guarantee procedure? gradient-function prim1)
  (lambda (scalar-parent)
    (guarantee scalar? scalar-parent prim1)
    (let ((parent-value (real-component scalar-parent)))
      ;; dual produced as output
      (dual (numeric-function parent-value)
            (lambda (scalar-self accumulator state-table)
              (check-link-input-types scalar-self accumulator state-table prim1)
              (let ((gradient-accumulation (gradient-function parent-value accumulator)))
                ((link-component scalar-parent) scalar-parent gradient-accumulation state-table)))))))

;; A helper procedure that helps define primitives for use in automatic
;; differentiation. It returns a procedure which takes in two duals as
;; arguments and produces a dual as output. The dual produced has a link
;; procedure that has access to the parents of this dual and a way of
;; accumulating gradients during the chain rule.
;; ----------------------------------------------------------
;; numeric-function: a function taking 2 numbers as input and
;;                   producing 1 number as output
;; gradient-function: a function used to accumulate gradient
;;                    results during chain rule. Describes how
;;                    the numeric-function should be combined
;;                    to produce the chain rule result. This
;;                    should produce 2 values
(define (prim2 numeric-function gradient-function)
  (guarantee procedure? numeric-function prim1)
  (guarantee procedure? gradient-function prim1)
  (lambda (scalar-parent1 scalar-parent2)
    (guarantee scalar? scalar-parent1 prim2)
    (guarantee scalar? scalar-parent2 prim2)
    (let ((parent-value1 (real-component scalar-parent1))
          (parent-value2 (real-component scalar-parent2)))
      ;; dual produced as output
      (dual (numeric-function parent-value1 parent-value2)
            (lambda (scalar-self accumulator state-table)
              (check-link-input-types scalar-self accumulator state-table prim2)
              (let-values (((gradient-accumulation1 gradient-accumulation2)
                            (gradient-function parent-value1 parent-value2 accumulator)))
                (let ((state-table-updated
                       ((link-component scalar-parent1) scalar-parent1 gradient-accumulation1 state-table)))
                  ((link-component scalar-parent2) scalar-parent2 gradient-accumulation2 state-table-updated))))))))

;;;========================================
;;; Definition of Procedures for use in
;;; automatic differentiation on duals
;;; These procedures do forward pass and
;;; create link procedure for backward pass
;;;========================================

;; run tests on inputs to ensure correct types for a gradient function used in prim1
(define (check-prim1-grad-func-types object accumulator caller)
  (guarantee number? object caller)
  (guarantee number? accumulator caller))

;; run tests on inputs to ensure correct types for a gradient function used in prim2
(define (check-prim2-grad-func-types object1 object2 accumulator caller)
  (guarantee number? object1 caller)
  (guarantee number? object2 caller)
  (guarantee number? accumulator caller))

;; produces procedure operating on single dual
(define 0:exp
  (prim1 s:exp
         (lambda (x accumulator)
           (check-prim1-grad-func-types x accumulator 0:exp)
           ;; f = exp(x), df/dx = exp(x) 
           ;; must chain df/dx with accumulator through multiplication
           ;; (df/dx * accumulator) = exp(x) * accumulator
           (s:* (s:exp x) accumulator))))

;; produces procedure operating on single dual
(define 0:sqrt
  (prim1 s:sqrt
         (lambda (x accumulator)
           (check-prim1-grad-func-types x accumulator 0:sqrt)
           ;; f = sqrt(x), df/dx = 1/(2 * sqrt(x))
           ;; must chain df/dx with accumulator through multiplication
           ;; (df/dx * accumulator) = 1/(2 * sqrt(x)) * accumulator
           (s:* (s:/ 1.0 (s:* 2 (s:sqrt x))) accumulator))))

;; produces procedure operating on single dual
(define 0:log
  (prim1 s:log
         (lambda (x accumulator)
           (check-prim1-grad-func-types x accumulator 0:log)
           ;; f = ln(x) = log(x), df/dx = 1/x
           ;; must chain df/dx with accumulator through multiplication
           ;; (df/dx * accumulator) = 1/x * accumulator
           (s:* (s:/ 1.0 x) accumulator))))

;; produces procedure operating on two duals
(define 0-0:+
  (prim2 s:+
         (lambda (x y accumulator)
           (check-prim2-grad-func-types x y accumulator 0-0:+)
           ;; f = x + y, df/dx = 1, df/dy = 1
           ;; must chain df/dx and df/dy with accumulator through multiplication
           ;; (df/dx * accumulator) = accumulator
           ;; (df/dy * accumulator) = accumulator
           (values accumulator accumulator))))

;; produces procedure operating on two duals
(define 0-0:-
  (prim2 s:-
         (lambda (x y accumulator)
           (check-prim2-grad-func-types x y accumulator 0-0:-)
           ;; f = x - y, df/dx = 1, df/dy = -1
           ;; must chain df/dx and df/dy with accumulator through multiplication
           ;; (df/dx * accumulator) = accumulator
           ;; (df/dy * accumulator) = -accumulator
           (values accumulator (s:- accumulator)))))

;; produces procedure operating on two duals
(define 0-0:/
  (prim2 s:/
         (lambda (x y accumulator)
           (check-prim2-grad-func-types x y accumulator 0-0:/)
           ;; f = x/y, df/dx = 1/y, df/dy = -x/(y^2)
           ;; must chain df/dx and df/dy with accumulator through multiplication
           ;; (df/dx * accumulator) = 1/y * accumulator
           ;; (df/dy * accumulator) = -x/(y^2) * accumulator
           (values (s:* (s:/ 1.0 y) accumulator) (s:* (s:/ (s:- x) (s:* y y)) accumulator)))))

;; produces procedure operating on two duals
(define 0-0:*
  (prim2 s:*
         (lambda (x y accumulator)
           (check-prim2-grad-func-types x y accumulator 0-0:*)
           ;; f = x * y, df/dx = y, df/dy = x
           ;; must chain df/dx and df/dy with accumulator through multiplication
           ;; (df/dx * accumulator) = y * accumulator
           ;; (df/dy * accumulator) = x * accumulator
           (values (s:* y accumulator) (s:* x accumulator)))))

;; produces procedure operating on two dual
(define 0-0:expt
  (prim2 s:expt
         (lambda (x y accumulator)
           (check-prim2-grad-func-types x y accumulator 0-0:expt)
           ;; f = expt(x, y), df/dx = y * expt(x, y-1), df/dy = ln(x) * expt(x, y) = log(x) * expt(x, y)
           ;; must chain df/dx with accumulator through multiplication
           ;; (df/dx * accumulator) = y * expt(x, y-1) * accumulator
           ;; (df/dy * accumulator) = log(x) * expt(x, y) * accumulator
           (values (s:* (s:* y (s:expt x (s:- y 1))) accumulator)
                   (s:* (s:* (s:log x) (s:expt x y)) accumulator)))))

;; helper for defining comparator functions
(define (comparator f)
  (lambda (x y)
    (guarantee scalar? x comparator)
    (guarantee scalar? y comparator)
    (f (real-component x) (real-component y))))

;; produces procedure operating on two duals
(define 0-0:<
  (comparator s:<))

;; produces procedure operating on two duals
(define 0-0:>
  (comparator s:>))

;; produces procedure operating on two duals
(define 0-0:<=
  (comparator s:<=))

;; produces procedure operating on two duals
(define 0-0:>=
  (comparator s:>=))

;; produces procedure operating on two duals
(define 0-0:=
  (comparator s:=))

;;;=============================================
;;; Tensors
;;;=============================================

(define tensor vector)

(define (tensor? object)
  (cond
   ((scalar? object) #t)
   ((vector? object) #t)
   (else
    #f)))

(define (tref tensor-object index)
  (guarantee tensor? tensor-object tref)
  (guarantee scalar? tensor-object tref)
  (vector-ref tensor-object (real-component index)))

(define (tlen tensor-object)
  (guarantee tensor? tensor-object tlen)
  (vector-length tensor-object))

(define (ref list-object scalar-object)
  (guarantee list? list-object ref)
  (guarantee scalar? scalar-object ref)
  (list-ref list-object (real-component scalar-object)))
