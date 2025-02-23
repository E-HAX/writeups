![alt text](image.png)

# Solution
We again get a godot game which has a timer. When the timer ends, we get the flag.

To bypass the timer, we can add our own logic for the timer in main.gd

```
var time_left = 172800
var simulated_time = 0
var simulation_speed = 100000

func _process(delta):
	simulated_time += delta * simulation_speed
	time_left = max(0, 172800 - simulated_time)
	$Label.text = str(int(time_left))
	update()
  ```

  ![alt text](image-1.png)