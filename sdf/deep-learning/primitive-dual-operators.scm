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
(define exp-0
  (prim1 s:exp
         (lambda (x accumulator)
           (check-prim1-grad-func-types x accumulator exp-0)
           ;; f = exp(x), df/dx = exp(x) 
           ;; must chain df/dx with accumulator through multiplication
           ;; (df/dx * accumulator) = exp(x) * accumulator
           (s:* (s:exp x) accumulator))))

;; produces procedure operating on single dual
(define sqrt-0
  (prim1 s:sqrt
         (lambda (x accumulator)
           (check-prim1-grad-func-types x accumulator sqrt-0)
           ;; f = sqrt(x), df/dx = 1/(2 * sqrt(x))
           ;; must chain df/dx with accumulator through multiplication
           ;; (df/dx * accumulator) = 1/(2 * sqrt(x)) * accumulator
           (s:* (s:/ 1.0 (s:* 2 (s:sqrt x))) accumulator))))

;; produces procedure operating on single dual
(define log-0
  (prim1 s:log
         (lambda (x accumulator)
           (check-prim1-grad-func-types x accumulator log-0)
           ;; f = ln(x) = log(x), df/dx = 1/x
           ;; must chain df/dx with accumulator through multiplication
           ;; (df/dx * accumulator) = 1/x * accumulator
           (s:* (s:/ 1.0 x) accumulator))))

;; produces procedure operating on two duals
(define +-0-0
  (prim2 s:+
         (lambda (x y accumulator)
           (check-prim2-grad-func-types x y accumulator +-0-0)
           ;; f = x + y, df/dx = 1, df/dy = 1
           ;; must chain df/dx and df/dy with accumulator through multiplication
           ;; (df/dx * accumulator) = accumulator
           ;; (df/dy * accumulator) = accumulator
           (values accumulator accumulator))))

;; produces procedure operating on two duals
(define --0-0
  (prim2 s:-
         (lambda (x y accumulator)
           (check-prim2-grad-func-types x y accumulator --0-0)
           ;; f = x - y, df/dx = 1, df/dy = -1
           ;; must chain df/dx and df/dy with accumulator through multiplication
           ;; (df/dx * accumulator) = accumulator
           ;; (df/dy * accumulator) = -accumulator
           (values accumulator (s:- accumulator)))))

;; produces procedure operating on two duals
(define /-0-0
  (prim2 s:/
         (lambda (x y accumulator)
           (check-prim2-grad-func-types x y accumulator /-0-0)
           ;; f = x/y, df/dx = 1/y, df/dy = -x/(y^2)
           ;; must chain df/dx and df/dy with accumulator through multiplication
           ;; (df/dx * accumulator) = 1/y * accumulator
           ;; (df/dy * accumulator) = -x/(y^2) * accumulator
           (values (s:* (s:/ 1.0 y) accumulator) (s:* (s:/ (s:- x) (s:* y y)) accumulator)))))

;; produces procedure operating on two duals
(define *-0-0
  (prim2 s:*
         (lambda (x y accumulator)
           (check-prim2-grad-func-types x y accumulator *-0-0)
           ;; f = x * y, df/dx = y, df/dy = x
           ;; must chain df/dx and df/dy with accumulator through multiplication
           ;; (df/dx * accumulator) = y * accumulator
           ;; (df/dy * accumulator) = x * accumulator
           (values (s:* y accumulator) (s:* x accumulator)))))

;; produces procedure operating on two dual
(define expt-0-0
  (prim2 s:expt
         (lambda (x y accumulator)
           (check-prim2-grad-func-types x y accumulator expt-0-0)
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
(define <-0-0
  (comparator s:<))

;; produces procedure operating on two duals
(define >-0-0
  (comparator s:>))

;; produces procedure operating on two duals
(define <=-0-0
  (comparator s:<=))

;; produces procedure operating on two duals
(define >=-0-0
  (comparator s:>=))

;; produces procedure operating on two duals
(define =-0-0
  (comparator s:=))
