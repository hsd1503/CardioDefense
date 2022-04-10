import numpy as np
from numpy.linalg import norm
import collections
from utils.smooth_signal import smooth
import torch

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Smooth function
def sm(data):
	data = data.squeeze()
	length = 9000
	temp1 = data[1:]
	temp0 = data[:length-1]
	diff = temp1 - temp0
	var = torch.std(diff)
	return var

class BoundaryAttack(object):
	def __init__(self, model, smooth_para, source_step, spherical_step, step_adaptation, win):
		super(BoundaryAttack, self).__init__()
		model.eval()
		self.model = model
		self.smooth_para = smooth_para
		self.source_step = source_step
		self.spherical_step = spherical_step
		self.stats_spherical_adversarial = collections.deque(maxlen=100)
		self.stats_step_adversarial = collections.deque(maxlen=30)
		self.type = np.float32
		self.step_adaptation = step_adaptation
		self.log_every_n_steps = 1
		self.win = win


	def prepare_generate_candidates(self, original, perturbed):
		unnormalized_source_direction = original - perturbed
		source_norm = torch.norm(unnormalized_source_direction)
		source_direction = unnormalized_source_direction / source_norm
		return unnormalized_source_direction, source_direction, source_norm
	

	def generate_candidate_default(self, original, unnormalized_source_direction, source_direction, source_norm):
		spherical_step = self.spherical_step
		source_step = self.source_step

		perturbation = np.random.randn(1,1,9000)

		shape = perturbation.shape
		# apply hanning filter
		if self.smooth_para:
			if self.win % 2 == 1:
				# Verifying Odd Window Size
				perturbation = smooth(perturbation.squeeze(), window_len=self.win)
				perturbation = perturbation.reshape(shape)


		# ===========================================================
		# calculate candidate on sphere
		# ===========================================================
		perturbation = torch.tensor(perturbation, dtype=torch.float32)
		perturbation = perturbation.to(device)
		dot = torch.sum(perturbation * source_direction)
		perturbation -= dot * source_direction
		perturbation *= spherical_step * source_norm / torch.norm(perturbation)

		D = 1 / np.sqrt(spherical_step ** 2 + 1)
		direction = perturbation - unnormalized_source_direction
		spherical_candidate = original + D * direction

		# ===========================================================
		# add perturbation in direction of source
		# ===========================================================

		new_source_direction = original - spherical_candidate
		new_source_direction_norm = torch.norm(new_source_direction)

		# length if spherical_candidate would be exactly on the sphere
		length = source_step * source_norm

		# length including correction for deviation from sphere
		deviation = new_source_direction_norm - source_norm
		length += deviation

		# make sure the step size is positive
		length = max(0, length)

		# normalize the length
		length = length / new_source_direction_norm

		candidate = spherical_candidate + length * new_source_direction
		
		
		return candidate, spherical_candidate

	def update_step_sizes(self):
		def is_full(deque):
			return len(deque) == deque.maxlen

		if not (is_full(self.stats_spherical_adversarial)
			or is_full(self.stats_step_adversarial)):
			# updated step size recently, not doing anything now
			return

		def estimate_probability(deque):
			if len(deque) == 0:
				return None
			return np.mean(deque)

		p_spherical = estimate_probability(self.stats_spherical_adversarial)
		p_step = estimate_probability(self.stats_step_adversarial)

		n_spherical = len(self.stats_spherical_adversarial)
		n_step = len(self.stats_step_adversarial)

		def log(message):
			_p_spherical = p_spherical
			if _p_spherical is None:  # pragma: no cover
				_p_spherical = -1.0

			_p_step = p_step
			if _p_step is None:
				_p_step = -1.0

			print(
				"  {} spherical {:.2f} ({:3d}), source {:.2f} ({:2d})".format(
					message, _p_spherical, n_spherical, _p_step, n_step
				)
			)

		if is_full(self.stats_spherical_adversarial):
			# Constrains orthogonal steps based on previous probabilities
			# where the spherical candidate is in the correct target class
			# so it's between .2 and .5 
			if p_spherical > 0.5:
				message = "Boundary too linear, increasing steps:	"
				self.spherical_step *= self.step_adaptation
				self.source_step *= self.step_adaptation
			elif p_spherical < 0.2:
				message = "Boundary too non-linear, decreasing steps:"
				self.spherical_step /= self.step_adaptation
				self.source_step /= self.step_adaptation
			else:
				message = None

			if message is not None:
				self.stats_spherical_adversarial.clear()
				log(message)

		if is_full(self.stats_step_adversarial):
			if p_step > 0.5:
				message = "Success rate too high, increasing source step:"
				self.source_step *= self.step_adaptation
			elif p_step < 0.2:
				message = "Success rate too low, decreasing source step: "
				self.source_step /= self.step_adaptation
			else:
				message = None

			if message is not None:
				self.stats_step_adversarial.clear()
				log(message)

	def attack(self, target_ecg, orig_ecg, orig, target, max_iterations, max_queries):
		# Added parameter queries to limit query amount, default is infinite
		self.model.eval()
		print("Initial spherical_step = {:.2f}, source_step = {:.2f}".format(
				self.spherical_step, self.source_step))
		print("Window size: {}".format(self.win))
		m = orig_ecg.shape[2]
		n_batches = 25
		perturbed = target_ecg
		original = orig_ecg
		query = 0
		iteration = 1
		query_dist_smooth = np.empty((3,1))
		# Iteration value for row 0
		query_dist_smooth[1][0] = torch.norm(original - perturbed)/m
		# Distance value for row 1
		query_dist_smooth[2][0] = sm(perturbed - original)
		bool = True
		# Set to false to terminate attack function
		rel_improvement_temp = 1
		new_perturbed = None

		while (iteration <= max_iterations):
						# Only appending every 10th step, not wasting memory
						# on every single step taken, rep. sample
			do_spherical = (iteration % 10 == 0)
			
			unnormalized_source_direction, source_direction, source_norm = self.prepare_generate_candidates(original, perturbed)
			distance = source_norm

			for i in range(n_batches):

				# generate candidates
				candidate, spherical_candidate = self.generate_candidate_default(original,
					unnormalized_source_direction, source_direction, source_norm)
				# candidate is the final result after both orthogonal and
				# source steps, while spherical step is just the
				# orthotogonal step wrt to the surface of the sphere
				if do_spherical:
					spherical_is_adversarial = (self.model(spherical_candidate).data.max(1, keepdim=True)[1] == target) # sphrical_can x_{o}^{i+1}
					query += 1
					if (query % 200 == 0):
						temp = np.empty((3,1))
						temp[0][0] = query / 100
						temp[1][0] = torch.norm(original - perturbed)/m
						temp[2][0] = sm(perturbed - original)
						query_dist_smooth = np.append(query_dist_smooth, temp, axis = 1)
					self.stats_spherical_adversarial.appendleft(spherical_is_adversarial.data.cpu())
					is_adversarial = (self.model(candidate).data.max(1, keepdim=True)[1] == target)
					self.stats_step_adversarial.appendleft(is_adversarial.data.cpu())
					# is_adversarial is wrt to the final position
					# and final perturbed image after both steps
					if is_adversarial:
						new_perturbed = candidate
						new_distance = torch.norm(new_perturbed - original)
						break
					# If final step is adversarial, then the for loops breaks
					# Sets the new_perturbed to not None, also updating the distance
				else:
					is_adversarial = (self.model(candidate).data.max(1, keepdim=True)[1] == target)
					query += 1
					if (query % 200 == 0):
						temp = np.empty((3,1))
						temp[0][0] = query / 100
						temp[1][0] = torch.norm(original - perturbed)/186
						temp[2][0] = sm(perturbed - original)
						query_dist_smooth = np.append(query_dist_smooth, temp, axis = 1)
					if is_adversarial:
						new_perturbed = candidate
						new_distance = torch.norm(new_perturbed - original)
						break

			message = ""
			if new_perturbed is not None:
				abs_improvement = distance - new_distance
				
				if abs_improvement > 0:
					rel_improvement = abs(abs_improvement) / distance
					message = "d. reduced by {:.3f}% ({:.4e})".format(
							rel_improvement * 100, abs_improvement
						)
					message += ' queries {}'.format(query)
					# update the variables
					perturbed = new_perturbed
					distance = new_distance

			iteration += 1
			self.update_step_sizes()

		if new_perturbed is not None:
			return perturbed, query
		else:
			return new_perturbed, query
