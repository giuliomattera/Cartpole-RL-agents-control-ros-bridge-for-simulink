#define NUMINPUT 4
#define NUMOUTPUT 1
#define ALPHA0 0.8
#define MU0 0.25
#define NUMHIDDENNEURONS 30
#define MAXEXE 100
// Setting global variables

// Define Matrix (architecture)
static float inputs[MAXEXE][NUMINPUT];
static float target[MAXEXE][NUMOUTPUT];

static float hidden_weights[NUMHIDDENNEURONS][NUMINPUT];
static float output_weights[NUMOUTPUT][NUMHIDDENNEURONS];
static float bh[NUMHIDDENNEURONS];
static float bo[NUMOUTPUT];

//Define variations
static float dwo[NUMOUTPUT][NUMHIDDENNEURONS];
static float dwh[NUMHIDDENNEURONS][NUMINPUT];
static float dbo[NUMOUTPUT];
static float dbh[NUMHIDDENNEURONS];

static int inp = NUMINPUT;
static int hid = NUMHIDDENNEURONS;
static int out = NUMOUTPUT;

static float x[NUMINPUT];
static float h[NUMHIDDENNEURONS];
static float y[NUMOUTPUT];


// Define Hyper-parameters

static float alpha = ALPHA0;
static float momentum = MU0;





