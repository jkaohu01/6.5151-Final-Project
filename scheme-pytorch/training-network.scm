


(define (split-into-groups lst n)
  (letrec ((split-helper (lambda (lst acc group count)
                           (cond
                             ((null? lst) (reverse (cons group acc)))
                             ((= count n) (split-helper lst (cons (reverse group) acc) '() 0))
                             (else (split-helper (cdr lst) acc (cons (car lst) group) (+ count 1)))))))
    (split-helper lst '() '() 0)))

(define (batch-dataset inputs outputs batch-size)
  (list (split-into-groups inputs batch-size) (split-into-groups outputs batch-size)))

      
(define (mse-loss preds outputs)
  (let ((difference (map (lambda (one-pred one-output)
                           (map n:+ one-pred
                                (map (lambda (cur-output)
                                                    (n:* cur-output -1)) one-output)))
                         preds outputs)))
    (let ((loss (map (lambda (one-set)
                       (fold-left n:+ (default-neuron 0.0) (map (lambda (diff1 diff2)
                                                                  (n:* diff1 diff2))
                                                                one-set one-set)))
                     difference)))
      (fold-left n:+ (default-neuron 0.0) loss))))
                  

(define (cross-entropy-loss preds outputs)
  (let ((product (map (lambda (one-pred one-output)
                        (map (lambda (pred output)
                               (n:-log (n:* pred output)))
                             one-pred one-output))
                      preds outputs)))
    (let ((loss (map (lambda (one-set)
                       (fold-left n:+ (default-neuron 0.0) one-set))
                     product)))
      (fold-left n:+ (default-neuron 0.0) loss))))

#|
(mse-loss (list (list (default-neuron 1.0) (default-neuron 10.0)) (list (default-neuron 3.0) (default-neuron 4.0))) (list (list (default-neuron 1.5) (default-neuron 9.0))(list (default-neuron 4.0) (default-neuron 1.0))))

(pp #@605)
#[neuron 605]
(value 11.25)
(grad 0.)
(backward #[compound-procedure 606])
(children (#[neuron 608] #[neuron 607]))
(operation +)
|#



(define (train-network network)
  (lambda (inputs outputs batch-size lr loss)
    (let ((batched-dataset (batch-dataset inputs outputs batch-size)))
      (let ((batched-inputs (car batched-dataset))
            (batched-outputs (cadr batched-dataset)))
        (let ((batched-preds (map (lambda (batch-input batch-output)
                                    (let ((batch-preds (map (lambda (item)
                                                              ((run-network network) item))
                                                            batch-input)))
                                      (let ((my-loss (loss batch-preds batch-output)))
                                        (n:backward my-loss)
                                        ((gradient-step network) lr)
                                        batch-preds)))
                                  batched-inputs batched-outputs)))
          (map (lambda (pred output) (pp (list (neuron-value (car (car pred))) (neuron-value (car (car output)))))) batched-preds batched-outputs)
          'done)))))

(define (sum elemList)
  (if
    (null? elemList)
    0
    (+ (car elemList) (sum (cdr elemList)))
  )
)

(define (normalize-one-input inputs)
  (map (lambda (item)
         (set-neuron-value! item (/ (neuron-value item) 255.0))
         item)
       inputs))

(define (normalize-pixel-inputs inputs)
  (map normalize-one-input inputs))

(define (round-to-binary value)
  (if (>= value 0.5) 1 0))

;; Doesn't seem to be working yet unfortunately
;; (define (test-binary-network network)
;;   (lambda (inputs outputs)
;;     (let ((network-preds (map (lambda (item) ((run-network network) item)) inputs)))
;;       (let ((correct (map (lambda (pred output)
;;                             (if (and (>= (neuron-value (car pred)) 0.5) (= (neuron-value (car output)) 1))
;;                                 1
;;                                 (if (and (< (neuron-value (car pred)) 0.5) (= (neuron-value (car output)) 0))
;;                                     1
;;                                     0)))
;;                           network-preds outputs)))
;;         (pp (map (lambda (neuron) (neuron-value (car neuron))) network-preds))
;;         (pp (map (lambda (neuron) (neuron-value (car neuron))) outputs))
;;         (pp correct)
;;         (pp (/ (sum correct) (length outputs)))))))



#|
Demo network:
- shows setup of network
- forward pass
- back prop
- gradient descent

(define a (fc-layer 2 2 #f))
(define b (fc-layer 2 1 #f))

(set-layer-weights! a (list (list (default-neuron 1.5) (default-neuron 3.0)) (list (default-neuron 2.2) (default-neuron 4.1))))
(set-layer-biases! a (list (default-neuron 0) (default-neuron 0)))
(set-layer-weights! b (list (list (default-neuron 3.1)) (list (default-neuron 1.6))))
(set-layer-biases! b (list (default-neuron 0)))

(define net (build-network (list a b)))

(define out (car ((run-network net) (list (default-neuron 2.0) (default-neuron 7.0)))))

(n:backward out)

((gradient-step net) 0.01))

|#


#|
Demo training loop with batching:

(define net (setup-network))
(define train-input (list (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))
      (list
       (default-neuron 2.0) (default-neuron 7.0))))
(define train-output (list (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))
      (list (default-neuron 10))))
((train-network net) train-input train-output 2 0.1)
|#
