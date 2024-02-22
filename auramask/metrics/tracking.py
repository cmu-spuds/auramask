from keras.metrics import Metric

class Tracking(Metric):
  def __init__(self, 
              name="track",
              **kwargs):
    super().__init__(name=name,**kwargs)
      
    self.total = self.add_weight(name='total', initializer='zeros')
      
  def get_config(self):
    return {
      "name": self.name,
    }

  def update_state(self, values):
    self.total.assign_add(values)
  
  def result(self):
    return self.total
  
  def reset_states(self):
    self.total.assign(0.)