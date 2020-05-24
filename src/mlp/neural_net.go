package main

import (
	"fmt"
	// "enums"
	m "matrix"
)

func main() {
	// var a enums.LayerType = "Activation"
	// fmt.Println(a)
	// Test matrix multiplication

	// Test 1
	var m1 *m.Matrix = m.MatrixFromArray( [][]float64{{ 1, 2 }, { 3, 4 }} )
	var m2 *m.Matrix = m.MatrixFromArray( [][]float64{{ 1, 0 }, { 0, 1 }} )
	var m1_by_m2, e = m.Multiply(m1, m2)
	if e != nil {
		fmt.Println(e)
	} else {
		fmt.Printf("M1:\n")
		m.PrintMatrix(m1)
		fmt.Printf("\nM2:\n")
		m.PrintMatrix(m2)
		fmt.Printf("\nM1 * M2:\n")
		m.PrintMatrix(&m1_by_m2)
		fmt.Print("\n\n")
	}


	// Test 2
	m1 = m.MatrixOfOnes(5, 5)
	m2 = m.MatrixOfZeros(5, 5)
	m1_by_m2, e = m.Multiply(m1, m2)
	if e != nil {
		fmt.Println(e)
	} else {
		fmt.Printf("M1:\n")
		m.PrintMatrix(m1)
		fmt.Printf("\nM2:\n")
		m.PrintMatrix(m2)
		fmt.Printf("\nM1 * M2:\n")
		m.PrintMatrix(&m1_by_m2)
		fmt.Print("\n\n")
	}

	// Test 3
	m1 = m.MatrixStdNorm(5, 5)
	m2 = m.MatrixIdentity(5, 5)
	m1_by_m2, e = m.Multiply(m1, m2)
	if e != nil {
		fmt.Println(e)
	} else {
		fmt.Printf("M1:\n")
		m.PrintMatrix(m1)
		fmt.Printf("\nM2:\n")
		m.PrintMatrix(m2)
		fmt.Printf("\nM1 * M2:\n")
		m.PrintMatrix(&m1_by_m2)
		fmt.Print("\n\n")
	}

	// Test 4
	m1 = m.MatrixStdNorm(5, 5)
	m2 = m.MatrixCopy(m1)
	m1_by_m2, e = m.Multiply(m1, m2)
	if e != nil {
		fmt.Println(e)
	} else {
		fmt.Printf("M1:\n")
		m.PrintMatrix(m1)
		fmt.Printf("\nM2:\n")
		m.PrintMatrix(m2)
		fmt.Printf("\nM1 * M2:\n")
		m.PrintMatrix(&m1_by_m2)
		fmt.Print("\n\n")
	}

}