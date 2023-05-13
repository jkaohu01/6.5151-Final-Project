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
(define s:sin (base-scheme 'sin))
(define s:cos (base-scheme 'cos))

;; Subtracts 1 from the numeric value
(define (sub1 numeric-value)
  (guarantee number? numeric-value sub1)
  (s:- numeric-value 1))

;; Adds 1 to the numeric value
(define (add1 numeric-value)
  (guarantee number? numeric-value add1)
  (s:+ numeric-value 1))
