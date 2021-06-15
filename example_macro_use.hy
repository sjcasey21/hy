(require [example-macro-def [m-ret]])

(defn foo []
  (m-ret 3)
  (print "This doesn't get printed."))

(print (foo))
