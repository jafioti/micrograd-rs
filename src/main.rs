mod value;
use value::*;

fn main() {
    let mut a = Value::new(2.0).label("a");
    println!("{}", a);
    let mut b = Value::new(-3.0).label("b");

    let mut c = Value::new(10.0).label("c");

    let mut e = (&mut a * &mut b).label("e");

    let mut d = (&mut c + &mut e).label("d");

    let mut f = Value::new(-2.0).label("f");

    let mut l = (&mut d * &mut f).label("L");

    let mut out = l.tanh().label("out");

    out.grad = 1.0;
    out.backward();

    out.graph();

    out.apply_grad(0.1);

    out.graph();
}
