use std::{fmt::Display, ops::{Add, Mul}};

use petgraph::{Graph, Directed, matrix_graph::NodeIndex, dot::{Dot, Config}};

#[derive(Debug)]
pub struct Value<'a> {
    pub value: f64,
    pub grad: f64,
    children: Option<Vec<&'a mut Value<'a>>>,
    operation: Option<Operation>,
    label: String,
}

#[derive(Debug, Clone, Copy)]
pub enum Operation {
    Add,
    Mul,
    Tanh
}

impl ToString for Operation {
    fn to_string(&self) -> String {
        match self {
            Operation::Add => "+",
            Operation::Mul => "-",
            Operation::Tanh => "tanh",
        }.to_string()
    }
}

impl <'a>Value<'a> {
    pub fn new(value: f64) -> Self {
        Self {
            value, 
            grad: 0.0,
            children: None,
            operation: None,
            label: "".to_string()
        }
    }

    fn children(mut self, children: Vec<&'a mut Value<'a>>) -> Self {
        self.children = Some(children);
        self
    }

    fn operation(mut self, operation: Operation) -> Self {
        self.operation = Some(operation);
        self
    }

   pub fn label<T: ToString>(mut self, label: T) -> Self {
        self.label = label.to_string();
        self
    }
}

impl <'a>Display for Value<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Value {{ label: {} data: {} grad: {}}}", self.label, self.value, self.grad)
    }
}

// Operators
impl <'a>Add<&'a mut Value<'a>> for &'a mut Value<'a> {
    type Output = Value<'a>;

    fn add(self, rhs: &'a mut Value<'a>) -> Self::Output {
        Value::new(self.value + rhs.value)
            .children(vec![self, rhs])
            .operation(Operation::Add)
    }
}

impl <'a>Mul<&'a mut Value<'a>> for &'a mut Value<'a> {
    type Output = Value<'a>;

    fn mul(self, rhs: &'a mut Value<'a>) -> Self::Output {
        Value::new(self.value * rhs.value)
            .children(vec![self, rhs])
            .operation(Operation::Mul)
    }
}

impl <'a>Value<'a> {
    pub fn tanh(&'a mut self) -> Value {
        Value::new(self.value.tanh())
            .children(vec![self])
            .operation(Operation::Tanh)
    }
}

// Graph
impl <'a>Value<'a> {
    pub fn graph(&self) {
        let mut graph = Graph::default();
        let start = graph.add_node(format!("{} | data: {} | grad: {}", self.label, self.value, self.grad));
        self.inner_graph(start, &mut graph);

        let url = format!("https://dreampuf.github.io/GraphvizOnline/#{}", urlencoding::encode(&Dot::with_config(&graph, &[Config::EdgeNoLabel]).to_string()));
        if let Err(e) = webbrowser::open(&url) {
            println!("Error displaying graph: {:?}", e);
        }
    }

    fn inner_graph(&self, curr_node: NodeIndex<u32>, graph: &mut Graph<String, bool, Directed, u32>) {
        if let Some(children) = &self.children {
            // Make new nodes for children
            let op_node = graph.add_node(self.operation.unwrap().to_string());
            let nodes: Vec<petgraph::stable_graph::NodeIndex> = children.iter()
                .map(|child| graph.add_node(format!("{} | data: {} | grad: {}", child.label, child.value, child.grad)))
                .collect();

            // Make edges
            graph.add_edge(op_node, curr_node, false);
            for node in &nodes {
                graph.add_edge(*node, op_node, false);
            }

            // Run on child nodes
            for (child, node) in children.iter().zip(nodes.into_iter()) {
                child.inner_graph(node, graph);
            }
        }
    }
}

// Backprop
impl <'a>Value<'a> {
    // Backprop gradients
    pub fn backward(&mut self) {
        if let Some(children) = &mut self.children {
            // Set child grads
            match self.operation.unwrap() {
                Operation::Add => {
                    for child in children.iter_mut() {
                        child.grad = self.grad;
                    }
                },
                Operation::Mul => {
                    // Assume there is only 2 children
                    children[0].grad = self.grad * children[1].value;
                    children[1].grad = self.grad * children[0].value;
                },
                Operation::Tanh => {
                    // Assume there is only 1 child
                    children[0].grad = (1.0 - self.value.powi(2)) * self.grad;
                }
            }

            // Propagate
            for child in children {
                child.backward();
            }
        }
    }

    // Apply gradients to values
    pub fn apply_grad(&mut self, learning_rate: f64) {
        self.value -= self.grad * learning_rate;
        
        if let Some(children) = &mut self.children {
            for child in children {
                child.apply_grad(learning_rate);
            }
        }
    }
}