;;; Tools for displaying error messages in pretty format.

(define (bad-arg-error arg-symbol procedure-symbol)
  (error "Bad argument"
         arg-symbol
         (error-irritant/noise " within procedure")
         procedure-symbol
         (error-irritant/noise ".")))

(define (or-bad-args-error arg1-symbol arg2-symbol procedure-symbol)
  (error "Bad argument"
         arg1-symbol
         (error-irritant/noise " or")
         arg2-symbol
         (error-irritant/noise " within procedure")
         procedure-symbol
         (error-irritant/noise ".")))

(define (and-bad-args-error arg1-symbol arg2-symbol procedure-symbol)
  (error "Bad arguments"
         arg1-symbol
         (error-irritant/noise " and")
         arg2-symbol
         (error-irritant/noise " within procedure")
         procedure-symbol
         (error-irritant/noise ".")))
