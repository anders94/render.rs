use crate::math::Matrix4;

pub struct TransformStack {
    stack: Vec<Matrix4>,
}

impl TransformStack {
    pub fn new() -> Self {
        Self {
            stack: vec![Matrix4::identity()],
        }
    }

    pub fn current(&self) -> Matrix4 {
        *self.stack.last().unwrap()
    }

    pub fn push(&mut self) {
        self.stack.push(self.current());
    }

    pub fn pop(&mut self) {
        if self.stack.len() > 1 {
            self.stack.pop();
        }
    }

    /// Replace the current transform (Identity / Transform requests).
    pub fn set(&mut self, transform: Matrix4) {
        *self.stack.last_mut().unwrap() = transform;
    }

    pub fn apply(&mut self, transform: Matrix4) {
        let current = self.current();
        let new = current * transform;
        *self.stack.last_mut().unwrap() = new;
    }
}

impl Default for TransformStack {
    fn default() -> Self {
        Self::new()
    }
}
