node {
  name: "model_input"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        unknown_rank: true
      }
    }
  }
}
node {
  name: "weight"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 1
          }
        }
        float_val: 2.003178
      }
    }
  }
}
node {
  name: "weight/read"
  op: "Identity"
  input: "weight"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@weight"
      }
    }
  }
}
node {
  name: "bias"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 1
          }
        }
        float_val: 0.009056054
      }
    }
  }
}
node {
  name: "bias/read"
  op: "Identity"
  input: "bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@bias"
      }
    }
  }
}
node {
  name: "Mul"
  op: "Mul"
  input: "model_input"
  input: "weight/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "add"
  op: "Add"
  input: "Mul"
  input: "bias/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
library {
}
