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

(function (global, factory) {
    let env;
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

    // Sum Function
    function Sum(begin, end, func = "x", sumType = "arithmetic") {
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
    const sigmoidDerivative = x => sigmoid(x) * (1 - sigmoid(x));
    const ReLU = x => Math.max(0, x);
    const softmax = (vector) => {
        const expVector = vector.map(v => Math.exp(v));
        const sumExp = expVector.reduce((a, b) => a + b, 0);
        return expVector.map(v => v / sumExp);
    };

    // Dual Numbers
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
    }

    // Pure Infinitesimal
    const epsilon = new DualNumber(0, 1);

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
    
    // Derivative (Finite Difference Approximation)
    function derivative(func, x, delta = 1e-5) {
        return (func(x + delta) - func(x - delta)) / (2 * delta);
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
                const diff = derivative(subFunc, 1); // Derivative at x=1 as a test
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
                    return `Substitution successful: ${newFunc.toString()}`;
                }
            } catch (e) {
                console.warn(`Error applying substitution for candidate: ${candidate}`);
            }
        }
    
        return `USubError: No matches for function ${func.toString()} found.`;
    }
    
    // Special Functions
    const exp = Math.exp;
    const erf = z => (2 / Math.sqrt(PI)) * DefiniteIntegral(t => exp(-t * t), 0, z);
    const gamma = z => DefiniteIntegral(t => Math.exp(-t) * Math.pow(t, z - 1), 0, Infinity);
    const Factorial = n => gamma(n + 1);
    const PolyLogarithm = (z, s, maxIterations = 1000) => {
        let sum = 0;
        for (let k = 1; k <= maxIterations; k++) {
            sum += Math.pow(z, k) / Math.pow(k, s);
        }
        return sum;
    };

    /* If integral is regularly uncomputable */
    function ImpossibleSubstitution(func) {
        // Find impossible part
        
    }
    
    /* Gradient */
    function Gradient(func, point) {
        return point.map((_, i) => PartialDerivative(func, point, i, point));
    }

    /* Perceptron and Neural Network Constructors */
    function Perceptron(inputs,weights,threshold=1.5){if(inputs.length!==weights.length){throw new Error("LengthError: Number of inputs must equal number of weights")}const weightedSum=inputs.reduce((sum,input,i)=>sum+input*weights[i],0);return[weightedSum>threshold,weightedSum]}
function BasicCreateNeuralNet(layerInfo,threshold=1.5){if(!Array.isArray(layerInfo)){throw new Error("Invalid layerInfo format. Must be an array.")}
const layerNumber=layerInfo[0];const numPerceptrons=layerInfo[1];const layers=layerInfo.map(([layerNumber,numPerceptrons])=>({layerNumber,perceptrons:Array.from({length:numPerceptrons},()=>({weights:initializeWeights(layerInfo[1]),threshold,})),}));return{layers,evaluate(inputs){return layers.reduce((outputs,layer)=>{return layer.perceptrons.map(p=>Perceptron(outputs,p.weights,p.threshold)[0])},inputs)},}}

    /* Gradient Descent */
    function GradientDescent(func, initialPoint, learningRate = 0.01, iterations = 100) {
        let point = [...initialPoint];

        for (let i = 0; i < iterations; i++) {
            const nabla = Gradient(func, point);
            point = point.map((x, j) => x - learningRate * nabla[j]);
        }

        return point;
    }

    /* Weight Initializer */
    function initializeWeights(numWeights, min = -1, max = 1) {
        return Array.from({ length: numWeights }, () => Math.random() * (max - min) + min);
    }

    // Exported API
    return {
        Sum,
        DualNumber,
        epsilon,
        Limit,
        Derivative,
        PartialDerivative,
        Integral,
        Gradient,
        Perceptron,
        BasicCreateNeuralNet,
        GradientDescent,
        initializeWeights,
        sigmoid,
        ReLU,
        softmax,
        erf,
        gamma,
        Factorial,
        PolyLogarithm
        
    };
});