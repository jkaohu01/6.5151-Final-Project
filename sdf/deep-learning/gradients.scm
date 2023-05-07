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
   ((vector? object) (begin
                       (vector-map (lambda (vector-element)
                                     (differentiable-map f vector-element))
                                   object)))
   (else
    (bad-arg-error 'object 'differentiable-map))))

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
   (else
    (bad-arg-error 'y 'gradient-state))))

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
