/**
 * Brahma.js
 * 
 * Brahma.js is a framework for working with math, statistics, calculus, and AI.
 * It is named after the Hindu god of creation, as it allows complex descriptions
 * of natural phenomena, due to its calculus and differential framework.
 * Unlike libraries such as Node.js and React.js, which use the '.js' suffix to
 * show that they are JavaScript frameworks, Brahma.js is an actual JS file, containing
 * all of this data. This is to make using and debugging the frameworks easy, 
 * as only one reference to one file is needed. Also, it has complex support for Node.js
 * and the Browswer/other enviornments.
*/

import { exec } from "child_process";
import { createServer } from "http";
import { app } from "electron";
import pkg from 'electron';
const { BrowserWindow } = pkg;

let env;
(function (global, factory) {
    if (typeof module === 'object' && typeof module.exports === 'object') {
        // Node.js environment
        module.exports = factory();
        env = "Node.js";
    } else {
        // Browser or other environments
        global.AI = factory();
        env = "Browser/other";
    }

    // Log the environment to verify
    console.log("Running in:", env);
})(this, function () {
    'use strict';

    /* Mathematical Tools */

    // Constants
    const e = Math.E;
    const PI = Math.PI;
    const π = PI;
    class Complex {
        constructor(real, imag) {
          this.real = real;
          this.imag = imag;
        }
      
        add(other) {
          return new Complex(this.real + other.real, this.imag + other.imag);
        }
      
        subtract(other) {
          return new Complex(this.real - other.real, this.imag - other.imag);
        }
      
        multiply(other) {
          const real = this.real * other.real - this.imag * other.imag;
          const imag = this.real * other.imag + this.imag * other.real;
          return new Complex(real, imag);
        }
      
        divide(other) {
          const denominator = other.real * other.real + other.imag * other.imag;
          const real = (this.real * other.real + this.imag * other.imag) / denominator;
          const imag = (this.imag * other.real - this.real * other.imag) / denominator;
          return new Complex(real, imag);
        }
      
        toString() {
          return `${this.real} + ${this.imag}i`;
        }
    }
    class Quaternion {
        constructor(real, i, j, k) {
            this.real = real;
            this.i = i;
            this.j = j;
            this.k = k;
        }
        toString() {
            return `${this.real} + ${this.i}i + ${this.j}j + ${this.k}k`;
        }
        add(other) {
            return new Quaternion(this.real + other.real, this.i + other.i, this.j + other.j, this.k + other.k);
        }
        subtract(other) {
            return new Quaternion(this.real - other.real, this.i - other.i, this.j - other.j, this.k - other.k);
        }
        multiply(other) {
            return new Quaternion(
                this.real * other.real - this.i * other.i - this.j * other.j - this.k * other.k,
                this.real * other.i + this.i * other.real + this.j * other.k - this.k * other.j,
                this.real * other.j + this.j * other.real + this.k * other.i - this.i * other.k,
                this.real * other.k + this.k * other.real + this.i * other.j - this.j * other.i
            );
        }
    }
    class Octonion {
        constructor(real, e1, e2, e3, e4, e5, e6, e7) {
            this.real = real;
            this.e1 = e1;
            this.e2 = e2;
            this.e3 = e3;
            this.e4 = e4;
            this.e5 = e5;
            this.e6 = e6;
            this.e7 = e7;
        }
        toString() {
            return `${this.real} + ${this.e1}e1 + ${this.e2}e2 + ${this.e3}e3 + ${this.e4}e4 + ${this.e5}e5 + ${this.e6}e6 + ${this.e7}e7`;
        }
        add(other) {
            return new Octonion(this.real + other.real, this.e1 + other.e1, this.e2 + other.e2, this.e3 + other.e3, this.e4 + other.e4, this.e5 + other.e5, this.e6 + other.e6, this.e7 + other.e7);
        }
        subtract(other) {
            return new Octonion(this.real - other.real, this.e1 - other.e1, this.e2 - other.e2, this.e3 - other.e3, this.e4 - other.e4, this.e5 - other.e5, this.e6 - other.e6, this.e7 - other.e7);
        }
        multiply(other) {
            return new Octonion(
                this.real * other.real - this.e1 * other.e1 - this.e2 * other.e2 - this.e3 * other.e3 - this.e4 * other.e4 - this.e5 * other.e5 - this.e6 * other.e6 - this.e7 * other.e7,  
                this.real * other.e1 + this.e1 * other.real + this.e2 * other.e7 - this.e3 * other.e6 + this.e4 * other.e5 - this.e5 * other.e4 + this.e6 * other.e3 - this.e7 * other.e2,
                this.real * other.e2 + this.e2 * other.real + this.e3 * other.e1 - this.e4 * other.e7 + this.e5 * other.e6 - this.e6 * other.e5 + this.e7 * other.e4,
                this.real * other.e3 + this.e3 * other.real + this.e4 * other.e1 - this.e5 * other.e7 + this.e6 * other.e6 - this.e7 * other.e5,
                this.real * other.e4 + this.e4 * other.real + this.e5 * other.e1 - this.e6 * other.e7 + this.e7 * other.e6,
                this.real * other.e5 + this.e5 * other.real + this.e6 * other.e1 - this.e7 * other.e7,
                this.real * other.e6 + this.e6 * other.real + this.e7 * other.e1,
                this.real * other.e7 + this.e7 * other.real
            );
        }
    }
    class GrassmanNumber {
        constructor(real, dual = 0) {
            this.real = real;
            this.dual = dual;
        }
        add(other) {
            return new GrassmanNumber(this.real + other.real, this.dual + other.dual);
        }
        multiply(other) {
            return new GrassmanNumber(
                this.real * other.real - this.dual * other.dual,
                this.real * other.dual + this.dual * other.real
            );
        }
        toString() {
            return `${this.real} + ${this.dual}θ`;
        }
    }
    class DualNumber {
        constructor(real, infinitesimal = 0) {
            this.real = real;
            this.infinitesimal = infinitesimal;
        }

        add(other) {
            return new DualNumber(
                this.real + other.real,
                this.infinitesimal + other.infinitesimal
            );
        }

        multiply(other) {
            return new DualNumber(
                this.real * other.real,
                this.real * other.infinitesimal + this.infinitesimal * other.real
            );
        }
        toString() {
            return `${this.real} + ${this.infinitesimal}ε`;
        }
    }
    const Supernumber = GrassmanNumber;
    const Anticommuting = GrassmanNumber;
    const i = new Complex(0, 1);
    const j = new Quaternion(0, 0, 1, 0);
    const k = new Quaternion(0, 0, 0, 1);
    const e0 = 1;
    const e1 = i;
    const e2 = j;
    const e3 = k;
    const e4 = new Octonion(0, 0, 0, 0, 1, 0, 0, 0);
    const e5 = new Octonion(0, 0, 0, 0, 0, 1, 0, 0);
    const e6 = new Octonion(0, 0, 0, 0, 0, 0, 1, 0);
    const e7 = new Octonion(0, 0, 0, 0, 0, 0, 0, 1);
    const iInfinity = i * Infinity;
    const epsilon = new DualNumber(0, 1);
    const theta = new GrassmanNumber(0, 1);
    const ε = epsilon;
    const θ = theta;
    const euler = Sum(1, Infinity, x => -ln(x) + 1 / x);   
    const pythagoras = sqrt(2);
    const golden = (1 + sqrt(5)) / 2;
    const phi = golden;
    const ϕ = phi;
    const goldenratio = phi;
    const apery = zeta(3);
    const bernouli = n => (((-1) ** n / 2 + 1) * 2 * Factorial(n) / (2 * PI) ** n) * zeta(n);
    const gelfond = exp(PI);
    const ramanajuan = exp(PI * sqrt(163));
    const hilbert = 2 ** pythagoras;
    
    // General Operation Function
    function Operation(a, b, operation) {
        operation = eval(operation);
        return operation(a, b);
    }

    // Sum Function
    function Sum(begin, end, func = x => x, sumType = "arithmetic") {
        if (Limit(func, Infinity) === Infinity) {
            function f(x) {
                if (x === 0) {
                    return func(begin);
                } 
                return func(x);
            }
            return generalzeta(f, -1); // Zeta Function Regularization (useful for computing divergent sums, for example 1+2+3+4+5+...=-1/12)
        }
        func = func.toString();
        if (end === Infinity) {
            end = 1e10;
        }
        if (begin === -Infinity) {
            begin = -1e10;
        }
        if (begin > end) {
            throw new Error("SumError: Begin must be less than or equal to End");
        }
        if (typeof begin !== "number" || typeof end !== "number") {
            throw new Error("SumError: Begin and End must be numbers");    
        }
        if (typeof func !== "string") {
            throw new Error("SumError: Function must be a string (i.e. 'x')");
        }
        const rule = new Function("x", `return ${func};`);
        let sum = 0;

        if (sumType === "arithmetic") {
            for (let i = begin; i <= end; i++) {
                sum += rule(i);
            }
            return sum;
        } else if (sumType === "geometric") {
            if (rule(begin) !== 1) {
                throw new Error("SumError: Beginning of Geometric Sum must equal 1");
            }
            const r = rule(begin + 1);
            const n = end - begin + 1;
            return (1 - r ** n) / (1 - r);
        } else {
            throw new Error(`Unsupported sumType '${sumType}' or invalid function '${func}' for sumType '${sumType}'.`);
        }
    }

    // Activation Functions
    const sigmoid = x => 1 / (1 + Math.exp(-x));
    const logisticCurve = sigmoid;
    const sigmoidDerivative = x => sigmoid(x) * (1 - sigmoid(x));
    const ReLU = x => Math.max(0, x);
    const softmax = (vector) => {
        const expVector = vector.map(v => exp(v));
        const sumExp = expVector.reduce((a, b) => a + b, 0);
        return expVector.map(v => v / sumExp);
    };

    // Statistics 
    const mean = dataset => Sum(0, dataset.length, `dataset[x]`, "arithmetic") / dataset.length;
    const average = mean;
    const median = dataset => {
        const len = dataset.length;
        if (len % 2 === 0) {
           return average([dataset[len / 2 - 1] + dataset[len / 2]]); 
        }
        return dataset[Math.floor(len / 2)];
    }
    const mode = dataset => {
        const counts = {};
        let maxCount = 0;
        let mode;
        for (const num of dataset) {
            counts[num] = (counts[num] || 0) + 1;
            if (counts[num] > maxCount) {
                maxCount = counts[num];
                mode = num;
            }
        }
        return mode;
    }
    const weightedmean = (dataset, weights) => dataset.length === weights.length ? Sum(1, dataset.length, x => dataset[x] * weights[x], "arithmetic") : new Error("Dataset and weights must have the same length");
    const samplestandardDeviation = dataset => sqrt(Sum(1, dataset.length, x => (dataset[x]-mean(dataset)) ** 2, 'arithmetic') / (dataset.length - 1));
    const samplevariance = dataset => samplestandardDeviation(dataset) ** 2;
    const standardDeviation = dataset => sqrt(Sum(1, dataset.length, x => (dataset[x]-mean(dataset)) ** 2, 'arithmetic') / dataset.length);
    const variance = dataset => standardDeviation(dataset) ** 2;
    const choose = (n, k) => Factorial(n) / ((Factorial(k) * Factorial(n - k)));
    const zvalue = (x, dataset) => (x - mean(dataset)) / standardDeviation(dataset);
    const zscore = zvalue;
    const correlationcoefficient = (xset, yset) => Sum(0, xset.length, x => (xset[x]*yset[x]) - (mean(xset)*mean(yset)), "arithmetic") / (sqrt(Sum(1, xset.length, 'xset[x] ** 2', "arithmetic") * Sum(1, yset.length, 'yset[x] ** 2', "arithmetic")));
    const r = correlationcoefficient;
    const GaussianPDF = (x, dataset) => 1 / (sqrt(2 * Math.PI) * variance(dataset)) * exp(-((x - mean(dataset)) ** 2) / (2 * sigma(dataset) ** 2)); 
    const ExpPDF = (x, parameter) => 1 - exp(-parameter * x) ? x >= 0 : 0;
    const UniformPDF = (x, a, b) => 1 / (b - a) ? a <= x && x <= b : 0;
    const ChiSquarePDF = (x, degreesOfFreedom) => 1 / (2 ** (degreesOfFreedom / 2) * gamma(degreesOfFreedom / 2)) * x ** (degreesOfFreedom / 2 - 1) * exp(-x / 2);
    const BetaPDF = (x, a, b) => (x ** (a - 1) * (1 - x) ** (b - 1)) / (Beta(a, b));
    const CDF = (x, PDF) => DefiniteIntegral(PDF, -Infinity, x);
    const ChiSquareCriticalValue = (chi, degreesOfFreedom) => {
        const EPS = 1e-12;
        let x = chi;
        let dx = 1000;
        while (Math.abs(dx) > EPS) {
            const p = ChiSquarePDF(x, degreesOfFreedom);
            dx = (p - 0.5) / (p * (x - (degreesOfFreedom - 2) / 2));
            x -= dx;
        }
        return x;
    }
    const MeanSquare = dataset => mean(dataset.map(x => x ** 2));

    // Statistical Tests
    function ChiSquare(observed, expected) {
        if (observed.length !== expected.length) {
            throw new Error("ChiSquareError: Observed and Expected must be the same length");
        }
        let chi = 0;
        for (let i = 0; i < observed.length; i++) {
            chi += (observed[i] - expected[i]) ** 2 / expected[i];
        }
        return chi;
    }
    const OneSampleZ = (sample, population) => (mean(sample) - mean(population)) / (standardDeviation(sample) / sqrt(sample.length));
    const OneSampleT = (sample, population) => (mean(sample) - mean(population)) / (samplestandardDeviation(sample) / sqrt(sample.length));
    const SpearmanRankCorrelationCoefficient = (differences, number_of_observations) => 1 - (6 * Sum(0, differences.length, `differences[x] ** 2`, "arithmetic")) / (number_of_observations * (number_of_observations ** 2 - 1));
    const SpearmanRank = SpearmanRankCorrelationCoefficient;
    
    // Limits
    function Limit(func, val, direction = null) {
        const forward = func(new DualNumber(val, epsilon.infinitesimal));
        const backward = func(new DualNumber(val, -epsilon.infinitesimal));

        if (direction === "+") {
            return forward.real;
        } else if (direction === "-") {
            return backward.real;
        } else if (forward.real === backward.real) {
            return forward.real;
        } else {
            throw new Error("LimitError: Limit does not exist at this point.");
        }
    }

    // Derivatives
    function Derivative(func, x, direction = null) {
        const h = epsilon.infinitesimal;
        const forward = (func(x + h).real - func(x).real) / h;
        const backward = (func(x).real - func(x - h).real) / h;

        if (direction === "+") {
            return forward;
        } else if (direction === "-") {
            return backward;
        } else if (forward === backward) {
            return forward;
        } else {
            throw new Error("DerivativeError: Derivative does not exist at this point.");
        }
    }
    function RepetitiveDerivative(func, x, numDerivatives = 1) {
        if (numDerivative === 0) {
            return func(x);
        }
        if (numDerivatives < 0) {
            throw new Error("DerivativeError: Number of derivatives must be non-negative.");
        }
        const h = epsilon.infinitesimal;
        let derivative = func(x);
        for (let i = 0; i < numDerivatives; i++) {
            derivative = (func(x + h).real - func(x).real) / h;
            x = derivative;
        }
        return derivative;
    }        
    function PartialDerivative(func, vars, varIndex, point) {
        const h = epsilon.infinitesimal;
        const newVars = [...vars];
        newVars[varIndex] += h;
        return (func(newVars).real - func(vars).real) / h;
    }

    /* Integral (methods to compute)*/
    
    // Definite Integral (Riemann Sum)
    function DefiniteIntegral(func, a, b, numIntervals = 1000, method = "trapezoidal") {
        // Handle Infinity (Approximation)
        const limit = 1e10; 
        if (a === -Infinity) a = -limit;
        if (b === Infinity) b = limit;
        if (a === -iInfinity) a = i * -limit;
        if (b === iInfinity) b = i * limit;
        if (a === Infinity) a = limit;
        if (b === -Infinity) b = -limit;

        const deltaX = (b - a) / numIntervals;
        let sum = 0;
    
        if (method === "trapezoidal") {
            for (let i = 0; i <= numIntervals; i++) {
                const x = a + i * deltaX;
                const weight = (i === 0 || i === numIntervals) ? 0.5 : 1; // Trapezoidal weights
                sum += weight * func(x);
            }
        } else if (method === "midpoint") {
            for (let i = 0; i < numIntervals; i++) {
                const x = a + (i + 0.5) * deltaX; // Midpoint of each subinterval
                sum += func(x);
            }
        }
        return sum * deltaX;
    }
    
    // Proportionality Check
    function isProportional(func1, func2) {
        try {
            const ratio = func1(1) / func2(1);
            return typeof ratio === "number" && ratio !== Infinity && ratio !== -Infinity;
        } catch {
            return false;
        }
    }
    
    // Decompose Function (Arrow or Traditional)
    function decomposeFunction(func) {
        const funcString = func.toString();
    
        // Detect if the function is an arrow function
        const isArrow = funcString.includes("=>");
    
        if (isArrow) {
            const [params, body] = funcString.split("=>").map(part => part.trim());
            return {
                parameters: params.replace(/^\((.*)\)$/, "$1").trim(), // Remove surrounding parentheses
                body,
            };
        } else {
            const params = funcString.match(/\(([^)]*)\)/)[1].trim();
            const body = funcString.match(/\{([\s\S]*)\}/)[1].trim();
            return { parameters: params, body };
        }
    }
    
    // Find Matches for Substitution
    function findMatches(func) {
        let matches = [];
        const decomposed = decomposeFunction(func);
    
        // Extract possible substitution candidates from function body
        const bodyParts = decomposed.body.split(/[\s;]+/); // Split by space or semicolon
        for (let sub of bodyParts) {
            try {
                const subFunc = new Function(decomposed.parameters, `return ${sub}`);
                const diff = Derivative(subFunc, 1); // Derivative at x=1 as a test
                if (isProportional(diff, func)) {
                    matches.push([sub, diff]); // Add to matches if proportional
                }
            } catch (e) {
                // Ignore invalid matches
                console.warn(`Skipping invalid sub-function: ${sub}`);
            }
        }
        return matches;
    }
    
    // Apply Substitution
    function ApplySub(func, replacement, location) {
        const funcString = func.toString();
        const updatedFuncString = funcString.replace(location, replacement);
        return new Function("x", updatedFuncString); // Recreate function from string
    }
    
    // u-Substitution
    function USub(func) {
        const candidates = findMatches(func);
    
        for (let [candidate, diff] of candidates) {
            // Find location of substitution in function string
            const funcString = func.toString();
            const location = funcString.indexOf(candidate);
    
            if (location === -1) continue; // Skip if candidate not found
    
            // Apply substitution
            const newFunc = ApplySub(func, "u", candidate);
    
            // Verify definite integral using substitution
            try {
                const result = DefiniteIntegral(x => eval(newFunc), 1, 2); // Replace with actual bounds
                if (typeof result === "number") {
                    return newFunc.toString();
                }
            } catch (e) {
                console.warn(`Error applying substitution for candidate: ${candidate}`);
            }
        }
    
        return `USubError: No matches for function ${func.toString()} found.`;
    }
    
    // Special Functions
    const exp = x => e ** x ? x !== NaN : cos(x) + i * sin(x);
    const ln = x => Math.log(x);
    const log = (x, base) => ln(x) / ln(base);
    const log10 = x => log(x, 10);
    const binarylog = x => log(x, 2);
    const sqrt = x => Math.sqrt(x);
    const sin = Math.sin;
    const cos = Math.cos;
    const tan = Math.tan;
    const sec = 1/ cos;
    const csc = 1/ sin;
    const cot = 1/ tan;
    const asin = Math.asin;
    const acos = Math.acos;
    const atan = Math.atan;
    const arcsin = asin;
    const arccos = acos;
    const arctan = atan;
    const sinh = Math.sinh;
    const cosh = Math.cosh;
    const tanh = Math.tanh;
    const sech = 1 / cosh;
    const csch = 1 / sinh;
    const coth = 1 / tanh;
    const arcsinh = Math.asinh;
    const arccosh = Math.acosh;
    const arctanh = Math.atanh;
    const erf = z => (2 / Math.sqrt(PI)) * DefiniteIntegral(t => exp(-t * t), 0, z);
    const erfi = x => -i * erf(x * i);
    const gamma = z => DefiniteIntegral(t => Math.exp(-t) * Math.pow(t, z - 1), 0, Infinity);
    const beta = (a, b) => gamma(a) * gamma(b) / gamma(a + b);
    const zeta = s => Sum(1, Infinity, x =>1 / (x ** s)) ? n > 1 : 2 ** s * PI ** (s - 1) * sin(PI * s / 2) * gamma(1 - s) * zeta(1 - s);
    const generalzeta = (a, s) => 1 / gamma(s) * DefiniteIntegral(t => t ** (s - 1) * Sum(n => exp(-a(n) * t), 0, Infinity), 0, Infinity);
    const Poisson = condition => condition ? 1 : 0;
    const delta = (i, j) => Poisson(i === j);
    const KroneckerDelta = delta;
    const LeviCivita = (indices) => {
      if (indices.length !== indices.length) return 0; // check for duplicate indices
      const sortedIndices = [...indices].sort((a, b) => a - b);
      let parity = 1;
      for (let i = 0; i < indices.length; i++) {
        const index = indices[i];
        const sortedIndex = sortedIndices[i];
        if (index !== sortedIndex) {
          parity *= -1;
          const swapCount = indices.slice(i).indexOf(sortedIndex);
          parity *= Math.pow(-1, swapCount);
        }
      }
      return parity;
    };
    const Factorial = n => gamma(n + 1);
    const PolyLogarithm = (z, s, maxIterations = 1000) => {
        let sum = 0;
        for (let k = 1; k <= maxIterations; k++) {
            sum += Math.pow(z, k) / Math.pow(k, s);
        }
        return sum;
    };
    const RisingFactorial = (a, n) => {
        let result = 1;
        for (let i = 1; i <= n; i++) {
            result *= (a + i - 1);
        }
        return result;
    };
    function GaussianHypergeometric(a_values, b_values, c_values, z) {
        for (let i of a_values) {
            for (let j of b_values) {
                for (let k of c_values) {
                    const function_to_be_iterated = n => ((RisingFactorial(i, n) * RisingFactorial(j, n)) / RisingFactorial(k, n)) * z ** n / Factorial(n);
                    return Sum(0, Infinity, function_to_be_iterated.toString(), "geometric");
                }
            }
        }
    };
    function Kummer(a, b, c, z) {
            const function_to_be_iterated = n => ((RisingFactorial(i, n) * RisingFactorial(j, n)) / RisingFactorial(k, n)) * z ** n / Factorial(n);
            return Sum(0, Infinity, function_to_be_iterated.toString(), "geometric");
    };
    function NewtonsMethod(func, x0, maxIterations = 1000) {
        let x = x0;
        for (let i = 0; i < maxIterations; i++) {
            x = x - func(x) / Derivative(func, x);
        }
        return x;
    }
    const root = NewtonsMethod;
    const LogarithmicIntegral = x => DefiniteIntegral(t => 1 / t, 0, x);
    const OffsetLI = x => LogarithmicIntegral(x) - LogarithmicIntegral(2); 
    const EllipticIntegral = (x, c, R, P) => DefiniteIntegral(t => R(t, sqrt(P(t))), c, x);
    function Power(base, exponent) {
        if (exponent === 0) return 1;
        if (exponent === 1) return base;
        if (exponent instanceof Complex) return Power(base ** exponent.real, exponent.imag ? (exponent.imag % 4 === 1 ? i : exponent.imag % 4 === 3 ? -i : exponent.imag % 4 === 2 ? -1 : 1) : 0);
        if (exponent instanceof DualNumber) return Power(base, exponent.real);
        if (exponent instanceof GrassmanNumber) return Power(base, exponent.real + exponent.dual * (1 - exponent.dual));
        return base * Power(base, exponent - 1);
    }
    function Tetration(base, number_of_iterations) {
        for (let i = 0; i < number_of_iterations; i++) {
            base = Power(base, base);
        }
        return base;
    }
    function Pentation(base, number_of_iterations) {
        for (let i = 0; i < number_of_iterations; i++) {
            base = Tetration(base, Power(base, base));
        }
        return base;
    }
    function Hexation(base, number_of_iterations) {
        for (let i = 0; i < number_of_iterations; i++) {
            base = Pentation(base, Power(base, base));
        }
        return base;
    }
    function ArrowNotation(base, number_of_arrows, number_of_iterations) {
        if (number_of_arrows === 1) return Power(base, number_of_iterations);
        for (let i = 0; i < number_of_iterations; i++) {
            base = ArrowNotation(base, number_of_arrows - 1, number_of_iterations);
        }
        return base;
    }
    const BinomialTheorem = (x, y, n) => Sum(0, n, k => choose(n, k) * Power(x, k) * Power(y, (n - k)));

    /* If integral is regularly uncomputable */
    function ImpossibleSubstitution(func) {
        // List of special functions to check
        const specialFunctions = [
            erf,
            erfi,
            gamma, 
            PolyLogarithm,
            LogarithmicIntegral,
            OffsetLI
        ];
    
        // Iterate over special functions to find a match
        for (let special of specialFunctions) {
            const { name, func: specialFunc } = special;
    
            // Scale factor to check match
            const scale = func(1) / specialFunc(1);
            if (!isNaN(scale) && Math.abs(scale - func(1) / specialFunc(1)) < 1e-6) {
                return x => scale * specialFunc(x);
            }
        }
    
        // If no match found, return null
        return null;
    }   

    // Find singularities
    function FindInfinite(func) {
        let x = 0;
        let y = 0;
        let hasSingularity = false;

        while (isFinite(func(x)) || isFinite(func(y))) {
            x += 1;
            y -= 1;
            
            // Check for singularity by attempting to evaluate func at nearby points
            if (!isFinite(func(x + 1e-9)) || !isFinite(func(y - 1e-9))) {
                hasSingularity = true;
                break;
            }
        }

        return hasSingularity ? [x, y] : null;
    }
    

    // Coordinate/Numerical Transformations
    const PolarToCartesian = (r, theta) => [r * cos(theta), r * sin(theta)];
    const CartesianToPolar = (x, y) => [sqrt(x ** 2 + y ** 2), atan(y / x)];
    const DegreesToRadians = x => x * PI / 180;
    const RadiansToDegrees = x => x * 180 / PI;

    // Expansions
    const taylorSeries = (func, x, a) => Sum(0, Infinity, n => RepetitiveDerivative(func, x, a) / Factorial(n) * (x - a) ** n, "arithmetic");
    const MacLaurinSeries = (func, x) => taylorSeries(func, x, 0);
    function FouirerSeries(func, x, period) {
        const L = period / 2;
        const a = n => 1 / L * DefiniteIntegral(m => func(m) * cos(n * PI * m / L), -L, L);
        const b = n => 1 / L * DefiniteIntegral(m => func(m) * sin(n * PI * m / L), -L, L);

        return a(0) + Sum(1, Infinity, n => a(n) * cos(n * PI * x / L) + b(n) * sin(n * PI * x / L));
    }

    // Integral Transformations
    function FourierTransform(func) {
        return omega => DefiniteIntegral(t => func(t) * exp(-2 * PI * i * omega * t), -Infinity, Infinity);
    }
    function LaplaceTransform(func) {
        return s => DefiniteIntegral(t => func(t) * exp(-s * t), 0, Infinity);
    }
    function InverseFourierTransform(func) {
        return t => DefiniteIntegral(omega => func(t) * exp(2 * PI * i * omega * t), -Infinity, Infinity);
    }
    function InverseLaplaceTransform(func) {
        let y = 0
        const inf = FindInfinite(func);
        if (inf) {
            y = inf[0] + 1;
        }

        return t => 1 / (2 * PI * i) * DefiniteIntegral(s => func(t) * exp(s * t), y - iInfinity, y + iInfinity);
    }
    
    /* Linear Algebra/Vector Calculus */

    // Unit vectors
    const ihat = [1, 0, 0, 0];
    const jhat = [0, 1, 0, 0];
    const khat = [0, 0, 1, 0];
    const Identity = size => {
        let I = [];
        for (let i = 0, j = 0; i < size && j < size; i++, j++) {
            I.push(delta(i, j));    
        }
        return I;
    }
    const ZeroMatrix = (m, n) => {
        let zero = [];
        for (let i = 0, j = 0; i < m && j < n; i++, j++) {
            zero.push(0);
        }  
    }
    // Vector Operations
    const AddMatrix = (v1, v2) => v1.map((_, i) => v1[i] + v2[i]);
    const SubtractMatrix = (v1, v2) => v1.map((_, i) => v1[i] - v2[i]);
    const ScaleMatrix = (v, a) => v.map((_, i) => v[i] * a);
    const MultiplyMatrix = (v1, v2) => v1.map((_, i) => v1[i] * v2[i]);
    const Transpose = v => v.map((_, i) => v.map((_, j) => v[j][i]));
    const magnitude = v => Math.sqrt(Sum(1, v.length, "v[i] ** 2"));
    const norm = magnitude;
    const dotproduct = (v1, v2) => Sum(1, v1.length, "v1[i] * v2[i]");
    const angle = (v1, v2) => arccos(dotproduct(v1, v2) / (magnitude(v1) * magnitude(v2)));
    const crossproduct = (v1, v2) => magnitude(v1) * magnitude(v2) * sin(angle(v1, v2));
    const determinate = v => v[0][0] * v[1][1] - v[0][1] * v[1][0];
    const KroneckerProduct = (A, B, lengthofA, widthofA) => {
        const p = [];
        for (let i = 0; i < lengthofA; i++) {
            for (let j = 0; j < widthofA; j++) {
                p.push(A[i][j] += B[i][j]);
            }
        }
        return sum;
    };


    // Gradient
    const Gradient = (func, point) => point.map((_, i) => PartialDerivative(func, point, i, point));

    // Divergence
    const Divergence = (func, point) => point.map((_, i) => PartialDerivative(func, point, i, point)).reduce((a, b) => a + b);

    // Curl
    const Curl = (func, point) => [
        PartialDerivative(func, point, 1, 2) - PartialDerivative(func, point, 2, 1),
        PartialDerivative(func, point, 2, 0) - PartialDerivative(func, point, 0, 2),
        PartialDerivative(func, point, 0, 1) - PartialDerivative(func, point, 1, 0),
    ];

    // Laplacian
    const Laplacian = (func, point) => Divergence(Gradient(func, point), point);

    // Hessian
    const HessianEntry = (func, point, i, j) => PartialDerivative(func, point, i, j);
    const Hessian = (func, point) => {
        const n = point.length;
        return [
            Array.from({length:n},(v,i)=>Array.from({length:n},(v2,j)=>HessianEntry(func,point,i,j)))
        ];
    };
    
    // Jacobian
    const Jacobian = (func, point) => {
        const n = point.length;
        return [
            Array.from({length:n},(v,i)=>PartialDerivative(func,point,i,point))
        ];
    }

    /* Plotting */
    
    class PlotFunc2d {
        constructor(func, title = null, loadInNewWindow = true, canvasName = "canvas", port = 8080) {
            this.func = func;
            this.title = title;
            this.port = port;
            this.loadInNewWindow = loadInNewWindow;
            this.canvasName = canvasName;
            
            if (env === "Node.js" && this.loadInNewWindow === false && this.port === null) {
                return "Error: When running in Node.js and loadInNewWindow is set to false, port must be specified";                
            }
            if (env === "Node.js") {
                this.plotNode();
            } else if (env === "Browser/other") {
                this.plotHTML();
            }
        }
        hasElectron() {
            try {
                require.resolve('electron');
                return true;
            } catch (e) {
                return false;
            }
        }
        plotNode() {
            if (this.loadInNewWindow) {
                if (!this.hasElectron()) {
                    exec("npm install electron");
                }
                let mainWindow;
                exec(`touch .index.html`);
                exec(`echo '<html><head><script src=\"../Brahma.js\"></script></head><body><script>const plot = new PlotFunc2d(${this.func}, ${this.title}, false, ${this.canvasName})).plotHTML();</script></body></html>' >> .index.html`);
                app.whenReady().then(() => {
                    mainWindow = new BrowserWindow({
                        width: 800,
                        height: 600,
                        webPreferences: {
                            nodeIntegration: true
                        }
                    });

                    mainWindow.loadFile('.index.html');
                });
                exec("rm .index.html");
            }
            createServer((_, res) => {
                res.writeHead(200, { 'Content-Type': 'text/html' });
                res.write(`<!doctypehtml><html lang=en><meta charset=UTF-8><meta content="width=device-width,initial-scale=1"name=viewport><title>${this.title}</title><style>canvas{border:1px solid #000}</style><h2>${this.title}</h2><button onclick=plotFunction()>Plot</button><br><br><canvas height=400 id=plotCanvas width=600></canvas><script>function plotFunction() {
                    const canvas = document.getElementById("plotCanvas");
                    const ctx = canvas.getContext("2d");
                    const funcInput = ${this.func};
                    
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    
                    // Draw axes
                    ctx.strokeStyle = "black";
                    ctx.beginPath();
                    ctx.moveTo(0, canvas.height / 2);
                    ctx.lineTo(canvas.width, canvas.height / 2);
                    ctx.moveTo(canvas.width / 2, 0);
                    ctx.lineTo(canvas.width / 2, canvas.height);
                    ctx.stroke();
                    
                    // Plot function
                    ctx.strokeStyle = "blue";
                    ctx.beginPath();
                    
                    const scaleX = 40; // Scale for x-axis
                    const scaleY = 40; // Scale for y-axis
                    
                    for (let i = -canvas.width / 2; i < canvas.width / 2; i++) {
                        let x = i / scaleX;
                        let y;
                        try {
                            y = eval(funcInput.replace(/x/g, (${x})));
                        } catch (e) {
                            console.error("Invalid function input", e);
                            return;
                        }
                        let canvasX = i + canvas.width / 2;
                        let canvasY = canvas.height / 2 - y * scaleY;
                        if (i === -canvas.width / 2) {
                            ctx.moveTo(canvasX, canvasY);
                        } else {
                            ctx.lineTo(canvasX, canvasY);
                        }
                    }
                    ctx.stroke();
                }
                
                plotFunction(); // Initial plot</script></html>');`);
                res.end();
            }).listen(this.port);
        }
        plotHTML() {
            const ctx = document.getElementById(this.canvasName).getContext("2d");
            const funcInput = this.func;
            const scaleX = 40;
            const scaleY = 40;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.strokeStyle = "black";
            ctx.beginPath();
            ctx.moveTo(0, canvas.height / 2);
            ctx.lineTo(canvas.width, canvas.height / 2);
            ctx.moveTo(canvas.width / 2, 0);
            ctx.lineTo(canvas.width / 2, canvas.height);
            ctx.stroke();
            ctx.strokeStyle = "blue";
            ctx.beginPath(); 
             
            for (let i = -canvas.width / 2; i < canvas.width / 2; i++) {
                let x = i / scaleX;
                let y = eval(funcInput.replace(/x/g, (x)));
                let canvasX = i + canvas.width / 2;
                let canvasY = canvas.height / 2 - y * scaleY;
                if (i === -canvas.width / 2) {
                    ctx.moveTo(canvasX, canvasY);
                } else {
                    ctx.lineTo(canvasX, canvasY);
                }
            }
            ctx.stroke();
        }
    }
    
    /* Artificial Intelligence */

    /* Perceptron and Neural Network Constructors */
    function Perceptron(inputs,weights,threshold=1.5){if(inputs.length!==weights.length){throw new Error("LengthError: Number of inputs must equal number of weights")}const weightedSum=inputs.reduce((sum,input,i)=>sum+input*weights[i],0);return[weightedSum>threshold,weightedSum]}function BasicCreateNeuralNet(layerInfo,threshold=1.5){if(!Array.isArray(layerInfo)){throw new Error("Invalid layerInfo format. Must be an array.")}const layers=layerInfo.map(([layerNumber,numPerceptrons])=>({layerNumber,perceptrons:Array.from({length:numPerceptrons},()=>({weights:initializeWeights(layerInfo[1]),threshold,})),}));return{layers,evaluate(inputs){return layers.reduce((outputs,layer)=>{return layer.perceptrons.map(p=>Perceptron(outputs,p.weights,p.threshold)[0])},inputs)},}}

    /* Gradient Descent */
    function GradientDescent(func, initialPoint, learningRate = 0.01, iterations = 100) {
        let point = [...initialPoint];

        for (let i = 0; i < iterations; i++) {
            const nabla = Gradient(func, point);
            point = point.map((x, j) => x  * -nabla[j]);
        }

        return point;
    }

    /* Weight Initializer */
    function initializeWeights(numWeights, min = -1, max = 1) {
        return Array.from({ length: numWeights }, () => Math.random() * (max - min) + min);
    }

    // Exported API
    return {
        env,
        e,
        PI,
        π,
        Complex,
        Quaternion,
        Octonion,
        DualNumber,
        GrassmanNumber,
        Supernumber,
        Anticommuting,
        i,
        j,
        k,
        e0,
        e1,
        e2,
        e3,
        e4,
        e5,
        e6,
        e7,
        iInfinity,
        epsilon,
        theta,
        ε,
        θ,
        euler,
        pythagoras,
        golden,
        phi,
        ϕ,
        goldenratio,
        apery,
        bernouli,
        gelfond,
        ramanajuan,
        hilbert,
        Operation,
        Sum,
        mean,
        average,
        median,
        mode,
        weightedmean,
        samplestandardDeviation,
        samplevariance,
        standardDeviation,
        variance,
        zvalue,
        zscore,
        correlationcoefficient,
        r,
        choose,
        OneSampleZ,
        OneSampleT,
        GaussianPDF,
        ExpPDF,
        ChiSquarePDF,
        BetaPDF,
        UniformPDF,
        CDF,
        Poisson,
        ChiSquareCriticalValue,
        ChiSquare,
        SpearmanRank,
        SpearmanRankCorrelationCoefficient,
        Limit,
        Derivative,
        PartialDerivative,
        DefiniteIntegral,
        ApplySub,
        USub,
        ImpossibleSubstitution,
        FindInfinite,
        PolarToCartesian,
        CartesianToPolar,
        DegreesToRadians,
        RadiansToDegrees,
        taylorSeries,
        MacLaurinSeries,
        FouirerSeries,
        FourierTransform,
        LaplaceTransform,
        InverseFourierTransform,
        InverseLaplaceTransform,
        Gradient,
        exp,
        ln,
        log,
        log10,
        binarylog,
        sqrt,
        sin,
        cos,
        tan,
        sec,
        csc,
        cot,
        asin,
        acos,
        atan,
        arcsin,
        arccos,
        arctan,
        sinh,
        cosh,
        tanh,
        sech,
        csch,
        coth,
        arcsinh,
        arccosh,
        arctanh,
        erf,
        erfi,
        gamma,
        beta,
        zeta,
        generalzeta,
        Poisson,
        delta,
        KroneckerDelta,
        LeviCivita,
        Factorial,
        PolyLogarithm,
        RisingFactorial,
        GaussianHypergeometric,
        Kummer,
        NewtonsMethod,
        root,
        LogarithmicIntegral,
        OffsetLI,
        EllipticIntegral,
        Power,
        Tetration,
        Pentation,
        Hexation,
        ArrowNotation,
        BinomialTheorem,
        ihat,
        jhat,
        khat,
        Identity,
        ZeroMatrix,
        AddMatrix,
        SubtractMatrix,
        ScaleMatrix,
        MultiplyMatrix,
        Transpose,
        magnitude,
        norm,
        dotproduct,
        angle,
        crossproduct,
        determinate,
        KroneckerProduct,
        Curl,
        Divergence,
        Laplacian,
        Gradient,
        HessianEntry,
        Hessian,
        Jacobian,
        PlotFunc2d,
        Perceptron,
        BasicCreateNeuralNet,
        GradientDescent,
        initializeWeights,
        sigmoid,
        logisticCurve,
        sigmoidDerivative,
        ReLU,
        softmax,
    };
})();