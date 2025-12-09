//go:build go1.1 && !go1.2 && !go1.3
// +build go1.1,!go1.2,!go1.3

package linear_models

import "C"

//export libLinearPrintFunc
func libLinearPrintFunc(str *C.char) {
	// Stubbed
}

func libLinearHookPrintFunc() {
	// Stubbed
}
