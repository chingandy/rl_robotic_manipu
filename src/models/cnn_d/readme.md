### Experiment Discription 

#### action space
``` action_range = [-0.1, -0.01, -0.005, 0.005, 0.01, 0.1] ```

#### Reward function

```python
 if dis < 0.05: # original setting: 0.008 => too strict
             plt.axis('off')
             plt.imshow(ob)
             plt.savefig('/home/chingan/thesis/rl_robotic_manipu/src/near_pic/img.png',transparent = True, bbox_inches = 'tight', pad_inches = 0)
             print("##########################Image saved.###########################")
             print("Near the target, distance: ", dis)
        if  dis <= 0.01:
            print("#"*50)
            print("Target touched!")
            print("#"*50)
            reward = 100
            self.rewards.append(reward)
            touch = True
            done = True
            return ob, reward, done, touch

        reward = np.exp(-gamma * dis)
        if len(self.rewards) < 3:
            pass
        elif np.all(self.rewards > reward):
            done = True
        else:
            done = False

        self.rewards.append(reward)
```
#### Model

```python
def build_model(self):
        # Architecture from the paper "3D simulation for robot arm control with deep q-learnin"
        # Define two sets of inputs
        main_input = Input(shape=inshape)
        aux_input = Input(shape=(2,))

        # The first branch operates on the main input
        x = Conv2D(64, kernel_size=5, strides=(2,2), activation='relu')(main_input)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Conv2D(32, kernel_size=5, strides=(2,2), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Conv2D(16, kernel_size=5, strides=(2,2), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Flatten()(x)

        # The second branch on the auxiliary input
        y = Dense(2, )(aux_input)

        # Combine the ouput of the two branches
        merged_vector = concatenate([x, y])

        # apply two FC layers and a regression prediction on the combined outputs
        output = Dense(4096, activation='relu')(merged_vector)
        output = Dense(256)(output)
        output = Dense(self.action_size)(output)
        model = Model(inputs=[main_input, aux_input], output=output)
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
       # plot_model(model, to_file='plot_model.png')
        return model

```
