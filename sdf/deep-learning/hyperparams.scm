;;;========================================
;;; File for Hyperparameters
;;;----------------------------------------
;;; When in use, this file should come before
;;; any use of the hyperparams defined here
;;; in the load-spec file so that is loaded
;;; before the dependent files
;;;========================================

#|
Example Template:

;; Hyperparam definitions
(define theta default-value-theta)
(define learning-rate default-value-learning-rate)
(define revs default-value-revs)

;; Then in the file using these hyperparams,
;; make sure you override the default values
;; by doing the following:
(let ((theta override-value-theta)
      (learning-rate override-value-learning-rate)
      (revs override-value-revs))
  <CODE BODY WHERE VALUES OVERRIDEN>)
|#
