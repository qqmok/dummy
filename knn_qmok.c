#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>



/* ========== qmok_login_dll.c ========== */
/* Double linked list from C-exercises. Author: Qiqi Mok, 2021 */


struct list_head {
    struct list_head *next, *prev;
};

/* initialize "shortcut links" for empty list */
void
list_init(struct list_head *head);

/* insert new entry after the specified head */
void 
list_add(struct list_head *new, struct list_head *head);

/* insert new entry before the specified head */
void 
list_add_tail(struct list_head *new, struct list_head *head);

/* deletes entry from list and reinitialize it, returns pointer to entry */
struct list_head* 
list_del(struct list_head *entry);

/* delete entry from one list and insert after the specified head */
void 
list_move(struct list_head *entry, struct list_head *head);

/* delete entry from one list and insert before the specified head */
void 
list_move_tail(struct list_head *entry, struct list_head *head);

/* tests whether a list is empty */
int 
list_empty(struct list_head *head);




/* initialize "shortcut links" for empty list */
void
list_init(struct list_head *head)
{
    head->next = head;
    head->prev = head;
}

/* insert new entry after the specified head */
void 
list_add(struct list_head *new, struct list_head *head)
{
    struct list_head *next_tmp = head;
    if (list_empty(head))
    {
        head->prev = new;
    }  
    else
    {
        next_tmp = head->next;
    }
    new->next = next_tmp;
    next_tmp->prev = new;
    new->prev = head;
    head->next = new;
}

/* insert new entry before the specified head */
void 
list_add_tail(struct list_head *new, struct list_head *head)
{
    struct list_head *prev_tmp = head;
    if (list_empty(head))
    {
        head->next = new;
    }  
    else
    {
        prev_tmp = head->prev;
    }
    new->next = head;
    new->prev = prev_tmp;
    prev_tmp->next = new;
    head->prev = new;
}

/* deletes entry from list and reinitialize it, returns pointer to entry */
struct list_head* 
list_del(struct list_head *entry)
{
    if (!list_empty(entry))
    {
        struct list_head *prev_tmp = entry->prev;
        struct list_head *next_tmp = entry->next;
        prev_tmp->next = next_tmp;
        next_tmp->prev = prev_tmp;
    }
    list_init(entry);
    return entry;
}

/* delete entry from one list and insert after the specified head */
void 
list_move(struct list_head *entry, struct list_head *head)
{
    list_add(list_del(entry), head);
}

/* delete entry from one list and insert before the specified head */
void 
list_move_tail(struct list_head *entry, struct list_head *head)
{
    list_add_tail(list_del(entry), head);
}

/* tests whether a list is empty */
int 
list_empty(struct list_head *head)
{
    return head->next == head && head->prev == head;
}


/* ========== end of qmok_login_dll.c ========== */



typedef struct {
  long index;
  double* coordinates;
  int class;
} Vector;


// Struct for containing information about neighbor points

typedef struct {
	struct list_head head;
  int class;
  double distance;
} neighbor_point_t;



// Struct for data point

typedef struct {
  int class;
  long block_index;
  double* coordinates;
  struct list_head* neighbors_head;
  int neighbors_counter;
  int* predictions;
  int neighbor_points_checked;
  pthread_mutex_t mutex;
  pthread_cond_t neighbor_cond;
  pthread_cond_t class_cond;
} test_point_t;


// Struct for containing information about k-nn class predictions

typedef struct {
  int k;
  long points_evaluated;
  long score;
  pthread_mutex_t mutex;
  pthread_cond_t evaluation_cond;
} knn_prediction_t;

typedef struct {
  struct list_head head;
  void* (*function)(void *);
  void* args;
} Task;



typedef struct {
  void* (*function)(void *);
  void* args;
  void* result;
} task_t;




typedef struct {
  int quit;
  int num_threads;
  pthread_t *threads;
  struct list_head *task_list;
  struct list_head *result_list;
  pthread_mutex_t queue_mutex;
  pthread_mutex_t result_mutex;
  pthread_cond_t queue_cond;
  pthread_cond_t result_cond;
} thread_pool_t;



Vector* data;
test_point_t* test_points;
long data_size;
int vector_size, class_size;
long N;
int k_max;
long B;
int n_threads;



void insert_new_neighbor(struct list_head *neighbors_head, neighbor_point_t *new_neighbor) {
	if (list_empty(neighbors_head)) {
		// puts("add neighbor to empty list");
		list_add((struct list_head*) new_neighbor, neighbors_head);
	}
	else {
		struct list_head *current_neighbor;
		current_neighbor = neighbors_head->next;
		for (int i = 0; i < k_max; i++) {
			if (current_neighbor == neighbors_head) {
				list_add_tail((struct list_head*) new_neighbor, current_neighbor);
				break;
			}
			if (new_neighbor->distance < ((neighbor_point_t*) current_neighbor)->distance) {
				// puts("insert neighbor before head");
				list_add_tail((struct list_head*) new_neighbor, current_neighbor);
				break;
			}
			current_neighbor = current_neighbor->next;
		}
	}
}

void free_neighbor_list(struct list_head *anker) {
	struct list_head *temp;
	temp = anker->next;
	while (temp != anker) {
		neighbor_point_t *temp_neighbor;
		temp_neighbor = (neighbor_point_t*) temp;
		temp = temp->next;
		free(temp_neighbor);
	}
	free(anker);
}

void free_test_points(test_point_t* test_points) {
	for (int i = 0; i < N; i++) {
		free_neighbor_list(test_points[i].neighbors_head);
		free(test_points[i].coordinates);
		free(test_points[i].predictions);
	}
	free(test_points);
}



void
print_list(struct list_head *start)
{
    puts("Printing list:");
    if (list_empty(start))
    {
        puts("The list is empty!");
    }
    else
    {
        struct list_head *tmp_item;
        tmp_item = start->next;
        while (tmp_item != start)
        {
            neighbor_point_t *tmp_proc = (neighbor_point_t*) tmp_item;
            printf("class:%d distance:%f   ", tmp_proc->class, tmp_proc->distance);
            tmp_item = tmp_item->next;
        }
        puts("\n");
    }
}



// compute distance between two test points
void *compute_distance(void *args) {
  void** pointers = (void**)args;
  test_point_t* test_point = (test_point_t*) pointers[1];
  pthread_mutex_lock(&test_point->mutex);
  for (int j = 0; j < N; j++) {
    if (test_points[j].block_index != test_point->block_index) { 
      double distance = 0.0;
      double* coord_test = test_point->coordinates;
      double* coord_target = test_points[j].coordinates;
    	
      for (int i = 0; i < vector_size; i++) {
        distance += pow(coord_test[i] - coord_target[i], 2);
      }
      neighbor_point_t *new_neighbor = malloc(sizeof(neighbor_point_t));
      new_neighbor->class = test_points[j].class;
      new_neighbor->distance = distance;
    	
      insert_new_neighbor(test_point->neighbors_head, new_neighbor);
      
    }
  }
  test_point->neighbor_points_checked = 1;
  pthread_mutex_unlock(&test_point->mutex);
  pthread_cond_broadcast(&test_point->neighbor_cond);
  
  free(args);
  
  return NULL;
}

// predict class for all k for a test point
void *predict_class(void *args) {

  void** pointers = (void**)args;
  test_point_t* test_point = (test_point_t*) pointers[0];
  pthread_mutex_lock(&test_point->mutex);
  while (test_point->neighbor_points_checked < 1) {
    pthread_cond_wait(&test_point->neighbor_cond, &test_point->mutex);
  }

	// predictions for test point
  long class_counter[class_size];
  for (int a = 0; a < class_size; a++) {
      class_counter[a] = 0;
  }
  int predicted_class = -1;
  long predicted_class_counter = 0;
  struct list_head *current_neighbor;
	current_neighbor = (test_point->neighbors_head)->next;
  for (int k = 0; k < k_max; k++) {  
    int current_neighbor_class = ((neighbor_point_t*) current_neighbor)->class;  
    class_counter[current_neighbor_class]++;
    current_neighbor = current_neighbor->next;
    if ((class_counter[current_neighbor_class] == predicted_class_counter && current_neighbor_class > predicted_class) 
      	|| class_counter[current_neighbor_class] > predicted_class_counter) {
      predicted_class_counter = class_counter[current_neighbor_class];
      predicted_class = current_neighbor_class;
    }
    (test_point->predictions)[k] = predicted_class;
  }
  pthread_mutex_unlock(&test_point->mutex);
  pthread_cond_broadcast(&test_point->class_cond);
  
  free(args);
  
  return NULL;
}

void *compute_score(void *args) {
  void** pointers = (void**)args;
  knn_prediction_t* knn = (knn_prediction_t*) pointers[0];
  test_point_t* test_point = (test_point_t*) pointers[1];
  pthread_mutex_lock(&knn->mutex);
  while ((test_point->predictions)[knn->k] < 0) {
    pthread_cond_wait(&test_point->class_cond, &knn->mutex);
  }
  if (test_point->class == (test_point->predictions)[(knn->k)-1]) {
    knn->score++;
  }
  knn->points_evaluated++;
  pthread_mutex_unlock(&knn->mutex);
  pthread_cond_signal(&knn->evaluation_cond);
  
  free(args);
  
  return NULL;
}


void *compute_quality(void *args) {
  void** pointers = (void**)args;
  knn_prediction_t* knn = (knn_prediction_t*) pointers[0];
  double* result = (double*) pointers[1];
  pthread_mutex_lock(&knn->mutex);
  while (knn->points_evaluated < N) {
    pthread_cond_wait(&knn->evaluation_cond, &knn->mutex);
  }
  pthread_mutex_unlock(&knn->mutex);
  double score = (double) knn->score / (double) N;
  result[(knn->k)-1] = score;
  
  free(args);
  
  return NULL;
}


void *perform_work(void *args) {
  thread_pool_t *thread_pool = (thread_pool_t*) args;
  while (thread_pool->quit == 0) {
    pthread_mutex_lock(&thread_pool->queue_mutex);
  	while (list_empty(thread_pool->task_list)) {
  		if (thread_pool->quit == 1) {
  			pthread_mutex_unlock(&thread_pool->queue_mutex);
  			return NULL;
  		}
  		pthread_cond_wait(&thread_pool->queue_cond, &thread_pool->queue_mutex);
  	}
    Task *task;
    task = (Task*) list_del((thread_pool->task_list)->next);
    pthread_mutex_unlock(&thread_pool->queue_mutex);
    task->function(task->args);
    pthread_mutex_lock(&thread_pool->result_mutex);
    list_add_tail((struct list_head*) task, thread_pool->result_list);
    pthread_mutex_unlock(&thread_pool->result_mutex);
    pthread_cond_signal(&thread_pool->result_cond);
  }
  return NULL;
}


void thread_pool_enqueue(thread_pool_t* thread_pool, void *(*function) (void *), void* args) {
  Task *new_task = malloc(sizeof(Task));
  new_task->function = function;
  new_task->args = args;
  
  pthread_mutex_lock(&thread_pool->queue_mutex);
  list_add_tail((struct list_head*) new_task, thread_pool->task_list);
  pthread_mutex_unlock(&thread_pool->queue_mutex);
  pthread_cond_signal(&thread_pool->queue_cond);
}

void thread_pool_init(thread_pool_t* thread_pool, int count) {
  thread_pool->task_list = malloc(sizeof(struct list_head));
  list_init(thread_pool->task_list);
  thread_pool->result_list = malloc(sizeof(struct list_head));
  list_init(thread_pool->result_list);
  thread_pool->queue_mutex = (pthread_mutex_t) PTHREAD_MUTEX_INITIALIZER;
  thread_pool->queue_cond = (pthread_cond_t) PTHREAD_COND_INITIALIZER;
  thread_pool->result_mutex = (pthread_mutex_t) PTHREAD_MUTEX_INITIALIZER;
  thread_pool->result_cond = (pthread_cond_t) PTHREAD_COND_INITIALIZER;
  
  thread_pool->threads = malloc(sizeof(pthread_t) * count);
  thread_pool->num_threads = count;
  for (int i = 0; i < count; i++) {
    if (pthread_create(&thread_pool->threads[i], NULL, perform_work, thread_pool) != 0) {
      puts("Creating thread failed");
    }
  }
}

Task* thread_pool_wait(thread_pool_t* thread_pool) {
  pthread_mutex_lock(&thread_pool->result_mutex);
  while (list_empty(thread_pool->result_list)) {
    pthread_cond_wait(&thread_pool->result_cond, &thread_pool->result_mutex);
  }
  Task *result = (Task*) list_del((thread_pool->result_list)->next);
  pthread_mutex_unlock(&thread_pool->result_mutex);
  return result;
}


void thread_pool_shutdown(thread_pool_t* thread_pool) {
  thread_pool->quit = 1;
  pthread_cond_broadcast(&thread_pool->queue_cond);
  for (int i = 0; i < thread_pool->num_threads; i++) {
    pthread_join(thread_pool->threads[i], NULL);
  }
  free(thread_pool);
}


void sequential(test_point_t* data) {
  long scores[k_max];
  for (int i = 0; i < k_max; i++) {
    scores[i] = 0;
  }
  
  for (int i = 0; i < N; i++) {
    double* coordinates_i = data[i].coordinates;
    int actual_class = data[i].class;
    
    int block_index = data[i].block_index;
    
    // neighbors & distances
    for (int j = 0; j < N; j++) {
      double distance = 0.0;
      double* coordinates_j = data[j].coordinates;
      if (data[j].block_index != block_index) {
      	// compute distance between point i and point j
        for (int a = 0; a < vector_size; a++) {
          distance += pow(coordinates_i[a] - coordinates_j[a], 2);
        }
        neighbor_point_t *new_neighbor = malloc(sizeof(neighbor_point_t));
        new_neighbor->distance = distance;
        new_neighbor->class = data[j].class;
        insert_new_neighbor(data[i].neighbors_head, new_neighbor);
      }
    }
    
    // predictions for test point i
    long class_counter[class_size];
    for (int a = 0; a < class_size; a++) {
        class_counter[a] = 0;
    }
    int predicted_class = -1;
    long predicted_class_counter = 0;
    struct list_head *current_neighbor;
		current_neighbor = data[i].neighbors_head->next;
    for (int k = 0; k < k_max; k++) {  
    	int current_neighbor_class = ((neighbor_point_t*) current_neighbor)->class;  
      class_counter[current_neighbor_class]++;
      current_neighbor = current_neighbor->next;
      if ((class_counter[current_neighbor_class] == predicted_class_counter && current_neighbor_class > predicted_class) 
      	   || class_counter[current_neighbor_class] > predicted_class_counter) {
        predicted_class_counter = class_counter[current_neighbor_class];
        predicted_class = current_neighbor_class;
      }
      
      if (predicted_class == actual_class) {
        scores[k]++;
      }
    }
  
  }
  
  double max_score = 0.0;
  int result_index = -1;
  for (int i = 0; i < k_max; i++) {
    double final_score = (double) scores[i] / (double) N;
    printf("%d %g\n", i, final_score);
    if (final_score >= max_score) {
      max_score = final_score;
      result_index = i;
    }
  }
  
  printf("%d\n", result_index);

}




int main(int argc, char* argv[]) {

  if (argc != 6) {
    puts("Please enter valid arguments.");
    return 0;
  }

  char* filename = argv[1];
  N = atol(argv[2]);
  k_max = atoi(argv[3]);
  B = atol(argv[4]);
  n_threads = atoi(argv[5]);
  
  FILE * fp = fopen(filename, "r");
  
  if (fscanf(fp, "%ld %d %d\n", &data_size, &vector_size, &class_size) == 0) {
    puts("Error occured while reading txt file.");
    return 0;
  }
  
  if (B < 2 || B > N) {
  	puts("B should be >1 and <N.");
  }
  
  if (data_size < N) {
    puts("The dataset size is smaller than N.");
    return 0;
  }
  
  // load data from file
  
  data = malloc(sizeof(Vector) * N);
  
  for (long i = 0; i < N; i++) {
    data[i].index = i;
    data[i].coordinates = malloc(sizeof(double) * vector_size);
    for (int j = 0; j < vector_size; j++) {
      if (fscanf(fp, "%lf ", &data[i].coordinates[j]) == 0) {
        puts("Error occured while reading coordinates.");
        return 1;
      }
    }
    if (fscanf(fp, "%d\n", &data[i].class) == 0) {
      puts("Error occured while reading class.");
      return 1;
    }
  }
  
  if (fclose(fp) != 0) {
    puts("Failed closing file.");
    return 1;
  }
  
  long block_size = N / B;
  test_points = malloc(sizeof(test_point_t) * N);
  
  for (long i = 0; i < N; i++) {
    pthread_mutex_t new_mutex;
    pthread_mutex_init(&new_mutex, NULL);
    pthread_cond_t new_neighbor_cond;
    pthread_cond_init(&new_neighbor_cond, NULL);
    pthread_cond_t new_class_cond;
    pthread_cond_init(&new_class_cond, NULL);
    test_point_t new_point = {
      .class= data[i].class,
      .block_index = (int) floor(i / block_size),
      .coordinates = malloc(sizeof(double) * vector_size),
      .neighbors_head = malloc(sizeof(struct list_head)),
      .neighbor_points_checked = 0,
      .predictions = malloc(sizeof(int) * k_max),
      .mutex = new_mutex,
      .neighbor_cond = new_neighbor_cond,
      .class_cond = new_class_cond
    };
    list_init(new_point.neighbors_head);	  
    for (int j = 0; j < vector_size; j++) {
      new_point.coordinates[j] = data[i].coordinates[j];
    }
    for (int j = 0; j < k_max; j++) {
    	new_point.predictions[j] = -1;
    }
    test_points[i] = new_point;
  } 
 
 
 
  knn_prediction_t predictions[k_max];
  // initalize knn predictions
  for (int i = 0; i < k_max; i++) {
    pthread_mutex_t new_mutex;
    pthread_mutex_init(&new_mutex, NULL);
    pthread_cond_t new_cond;
    pthread_cond_init(&new_cond, NULL);
    knn_prediction_t new_prediction = {
      .k = i+1,
      .points_evaluated = (long) 0,
      .mutex = new_mutex,
      .evaluation_cond = new_cond,
    };
    predictions[i] = new_prediction;
  }
  
  
  if (n_threads == 0) {
    sequential(test_points);
  }


  else if (n_threads > 0) {
  
    // create and initialize thread pool
    thread_pool_t* thread_pool;
    thread_pool = malloc(sizeof(thread_pool_t));
    thread_pool_init(thread_pool, n_threads);
    
    
    
    // part 1
    for (long i = 0; i < N; i++) {
    	void** args = malloc(sizeof(void*) * 2);
      args[0] = (void*) &test_points;
      args[1] = (void*) &test_points[i];
      thread_pool_enqueue(thread_pool, compute_distance, args);
      // free(thread_pool_wait(thread_pool));
    }

  
    // part 2
    for (int k = 0; k < k_max; k++) {
      for (int i = 0; i < N; i++) {
        void** args = malloc(sizeof(void*) * 2);
        args[1] = (void*) &predictions[k];
        args[0] = (void*) &test_points[i];
        thread_pool_enqueue(thread_pool, predict_class, args);
        // free(thread_pool_wait(thread_pool));
      }
    }


    // part 3
    for (int k = 0; k < k_max; k++) {
      for (int i = 0; i < N; i++) {
        void** args = malloc(sizeof(void*) * 2);
        args[0] = (void*) &predictions[k];
        args[1] = (void*) &test_points[i];
        thread_pool_enqueue(thread_pool, compute_score, args);
        // free(thread_pool_wait(thread_pool));
      }
    }
  

    double result_scores[k_max];
    for (int k = 0; k < k_max; k++) {
      void** args = malloc(sizeof(void*) * 2);
      args[0] = (void*) &predictions[k];
      args[1] = (void*) result_scores;
      thread_pool_enqueue(thread_pool, compute_quality, args);
      // free(thread_pool_wait(thread_pool));
    }
  	

    while(!list_empty(thread_pool->task_list)) {
      free(thread_pool_wait(thread_pool));
    }
    

    int result_k = -1;
    double result_score = 0.0;
    for(int k = 0; k < k_max; k++) {
      printf("%d %g\n", k, result_scores[k]);
      if (result_scores[k] >= result_score) {
        result_k = k;
        result_score = result_scores[k];
      }
    }
    printf("%d\n", result_k);
  	

    thread_pool_shutdown(thread_pool);
    
  
  }
  
  
  free_test_points(test_points);
  
  return 0;
}
