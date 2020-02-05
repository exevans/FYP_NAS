train.py

Creates a StateSpace
adds states kernel and filters each with acceptable values
Print the state space()

Create a controller(num_layers, state_space)
Create network manager
get initial random state space
print initial state

TRAINING BEGINS
	
for max number of models to generate



StateSpace
	add_state(name value)
		for all values for a param
			add them to index_map
			add i to value_map

			store metadata
				get the state id (num of variables added)
				name
				values (all of them as a list)
				size (num values)
				index_map
				value_map
			metadata stored at self.states[state id]
			inc states_count

	print_state_space
		print metadata for each state

	get_random_state_space <- random initial state to feed controller
		for id in range (state_num*num_layers) states are params so we need to choose one for all params all layers
			get the state with id
			get num of values
			sample = choose a random one
			set the state as self.embedding_encode(id, sample)

			add to defined states the encoded version
			return the states

	embedding_encode(state id, value to use)
		get the state to encode
		get the number of possible values
		get value_map 
		get value_index = the index of the chosen value

		do a one hot encode = array of 0's of height size (number of possible values')
		at index that equals chosen value set as the index of chosen val + 1


	size is the number of states

	getitem []
		returns the state equal to id % size
		--eq the param regardless of layer