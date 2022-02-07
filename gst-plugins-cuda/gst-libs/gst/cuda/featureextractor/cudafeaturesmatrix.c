/**************************** Includes and Macros *****************************/

#include <gst/cuda/featureextractor/cudafeaturescell.h>
#include <gst/cuda/featureextractor/cudafeaturesmatrix.h>

#include <glib-object.h>

/************************** Type/Struct Definitions ***************************/

/*************************** Function Declarations ****************************/

/**
 * \brief Callback function for freeing entries within the gpointerarray
 * instance.
 *
 * \details This callback function is used when the gpointerarray instance held
 * by an instance of the CUDAFeaturesMatrix GObject unref's the gpointerarray.
 * Each individual entry within the gpointerarray is checked to make certain it
 * is a CUDAFeaturesCell GObject instance and is unreferenced.
 *
 * \param[in,out] data The current entry within the gpointerarray instance that
 * has been passed to this callback function.
 */
static void cuda_features_matrix_cell_free(gpointer data);

/**
 * \brief Constructed callback method for the CUDAFeaturesMatrix GObject.
 *
 * \details Upon the successful creation of a CUDAFeaturesMatrix GObject
 * instance, this method is called to provision an array of CUDAFeaturesCell
 * instances. The references to each indivdiual entry of the CUDAFeaturesCell
 * array is held within a pointer array on the CUDAFeaturesMatrix GObject
 * instance.
 *
 * \param[in,out] object A CUDAFeaturesMatrix GObject instance to create and
 * store all the CUDAFeaturesCell GObject instance references on.
 */
static void cuda_features_matrix_constructed(GObject *object);

/**
 * \brief Finaliser method for the CUDAFeaturesMatrix GObject.
 *
 * \details Upon the destruction of an instance of the CUDAFeaturesMatrix
 * GObject, this method is called to clean-up the array of references to the
 * CUDAFeaturesCell instances created by this instance of the
 * CUDAFeaturesMatrix GObject. Unlike the dispose method, the finalise method
 * is only ever called once.
 *
 * \param[in,out] object A CUDAFeaturesMatrix GObject instance to release all
 * held resources from.
 */
static void cuda_features_matrix_finalize(GObject *object);

/**
 * \brief Property getter for instances of the CUDAFeaturesMatrix GObject.
 *
 * \param[in] object A CUDAFeaturesMatrix GObject instance to get the property from.
 * \param[in] prop_id The ID number for the property to get. This should
 * correspond to an entry in the enum defined in this file.
 * \param[out] value The GValue instance (generic value container) that will
 * contain the value of the property after being converted into the generic
 * GValue type.
 * \param[in] pspec The property specification instance for the property that
 * we are getting. This property specification should exist on the
 * CUDAFeaturesMatrixClass GObjectClass structure.
 *
 * \notes Any property ID that does not exist on the CUDAFeaturesMatrixClass
 * GClassObject will result in that property being ignored by the getter.
 */
static void cuda_features_matrix_get_property(
    GObject *object,
    guint prop_id,
    GValue *value,
    GParamSpec *pspec);

/**
 * \brief Property setter for instances of the CUDAFeaturesMatrix GObject.
 *
 * \param[in,out] gobject A CUDAFeaturesMatrix GObject instance to set the
 * property on.
 * \param[in] prop_id The ID number for the property to set. This should
 * correspond to an entry in the enum defined in this file.
 * \param[in] value The GValue instance (generic value container) that will be
 * convered to the correct value type to set the property with.
 * \param[in] pspec The property specification instance for the property that
 * we are setting. This property specification should exist on the
 * CUDAFeaturesMatrixClass GObjectClass structure.
 *
 * \notes Any property ID that does not exist on the CUDAFeaturesMatrixClass
 * GClassObject will result in that property being ignored by the setter.
 */
static void cuda_features_matrix_set_property(
    GObject *object,
    guint prop_id,
    const GValue *value,
    GParamSpec *pspec);

/****************************** Static Variables ******************************/

/**
 * \brief Anonymous enumeration containing the list of properties available for
 * the CUDAFeaturesMatrix GObject type this module defines.
 */
enum
{
    /**
     * \brief ID number for the FeatureMatrixHeight feature property.
     */
    PROP_FEATURES_MATRIX_ROWS = 1,

    /**
     * \brief ID number for the FeatureMatrixWidth feature property.
     */
    PROP_FEATURES_MATRIX_COLS,

    /**
     * \brief Number of property ID numbers in this enum.
     */
    N_PROPERTIES
};

static const guint32 default_feature_matrix_cols = 20u;
static const guint32 default_feature_matrix_rows = 20u;

/************************** GObject Type Definitions **************************/

G_DEFINE_TYPE_WITH_PRIVATE(
    CUDAFeaturesMatrix,
    cuda_features_matrix,
    G_TYPE_OBJECT);

/**************************** Function Definitions ****************************/

CUDAFeaturesCell *
cuda_features_matrix_at(CUDAFeaturesMatrix *self, guint32 col, guint32 row)
{
    CUDAFeaturesCell *cell = NULL;
    guint32 offset = 0;
    CUDAFeaturesMatrixPrivate *self_private = NULL;

    self_private = (CUDAFeaturesMatrixPrivate
                        *)(cuda_features_matrix_get_instance_private(self));

    g_return_val_if_fail(col < self_private->features_matrix_cols, NULL);
    g_return_val_if_fail(row < self_private->features_matrix_rows, NULL);

    offset = (row * self_private->features_matrix_cols) + col;

    cell = CUDA_FEATURES_CELL(
        g_ptr_array_index(self_private->features_matrix, offset));

    return CUDA_FEATURES_CELL(g_object_ref(G_OBJECT(cell)));
}

static void cuda_features_matrix_cell_free(gpointer data)
{
    if(data != NULL && G_IS_OBJECT(data))
    {
        g_object_unref(data);
    }
}

static void cuda_features_matrix_class_init(CUDAFeaturesMatrixClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    GParamSpec *properties[N_PROPERTIES] = {
        NULL,
    };

    gobject_class->constructed = cuda_features_matrix_constructed;
    gobject_class->finalize = cuda_features_matrix_finalize;
    gobject_class->get_property = cuda_features_matrix_get_property;
    gobject_class->set_property = cuda_features_matrix_set_property;

    properties[PROP_FEATURES_MATRIX_COLS] = g_param_spec_uint(
        "features-matrix-cols",
        "Features Matrix Columns",
        "The number of columns for the features matrix.",
        0,
        G_MAXUINT,
        default_feature_matrix_cols,
        (GParamFlags)(G_PARAM_READWRITE | G_PARAM_CONSTRUCT_ONLY | G_PARAM_STATIC_STRINGS));

    properties[PROP_FEATURES_MATRIX_ROWS] = g_param_spec_uint(
        "features-matrix-rows",
        "Features Matrix Rows",
        "The number of rows for the features matrix.",
        0,
        G_MAXUINT,
        default_feature_matrix_rows,
        (GParamFlags)(G_PARAM_READWRITE | G_PARAM_CONSTRUCT_ONLY | G_PARAM_STATIC_STRINGS));

    g_object_class_install_properties(gobject_class, N_PROPERTIES, properties);
}

static void cuda_features_matrix_constructed(GObject *object)
{
    CUDAFeaturesMatrix *self = CUDA_FEATURES_MATRIX(object);
    CUDAFeaturesMatrixPrivate *self_private
        = (CUDAFeaturesMatrixPrivate
               *)(cuda_features_matrix_get_instance_private(self));

    self_private->features_matrix
        = g_ptr_array_new_with_free_func(cuda_features_matrix_cell_free);

    g_ptr_array_set_size(
        self_private->features_matrix,
        self_private->features_matrix_cols
            * self_private->features_matrix_rows);

    for(guint32 row = 0; row < self_private->features_matrix_rows; row++)
    {
        for(guint32 col = 0; col < self_private->features_matrix_cols; col++)
        {
            CUDAFeaturesCell *cell
                = g_object_new(CUDA_TYPE_FEATURES_CELL, NULL);
            guint32 offset = (row * self_private->features_matrix_cols) + col;

            g_ptr_array_index(self_private->features_matrix, offset) = cell;
        }
    }

    if(G_OBJECT_CLASS(cuda_features_matrix_parent_class)->constructed != NULL)
    {
        G_OBJECT_CLASS(cuda_features_matrix_parent_class)->constructed(object);
    }
}

static void cuda_features_matrix_finalize(GObject *object)
{
    CUDAFeaturesMatrix *self = CUDA_FEATURES_MATRIX(object);
    CUDAFeaturesMatrixPrivate *self_private
        = (CUDAFeaturesMatrixPrivate
               *)(cuda_features_matrix_get_instance_private(self));

    if(self_private->features_matrix != NULL)
    {
        g_ptr_array_free(self_private->features_matrix, TRUE);
    }

    if(G_OBJECT_CLASS(cuda_features_matrix_parent_class)->finalize != NULL)
    {
        G_OBJECT_CLASS(cuda_features_matrix_parent_class)->finalize(object);
    }
}

static void cuda_features_matrix_get_property(
    GObject *object,
    guint prop_id,
    GValue *value,
    GParamSpec *pspec)
{
    CUDAFeaturesMatrix *self = CUDA_FEATURES_MATRIX(object);
    CUDAFeaturesMatrixPrivate *self_private
        = (CUDAFeaturesMatrixPrivate
               *)(cuda_features_matrix_get_instance_private(self));

    switch(prop_id)
    {
        case PROP_FEATURES_MATRIX_COLS:
            g_value_set_uint(value, self_private->features_matrix_cols);
            break;
        case PROP_FEATURES_MATRIX_ROWS:
            g_value_set_uint(value, self_private->features_matrix_rows);
            break;
        default:
            g_assert_not_reached();
    }
}

static void cuda_features_matrix_init(CUDAFeaturesMatrix *self)
{
    CUDAFeaturesMatrixPrivate *self_private
        = (CUDAFeaturesMatrixPrivate
               *)(cuda_features_matrix_get_instance_private(self));

    self_private->features_matrix = NULL;
}

static void cuda_features_matrix_set_property(
    GObject *object,
    guint prop_id,
    const GValue *value,
    GParamSpec *pspec)
{
    CUDAFeaturesMatrix *self = CUDA_FEATURES_MATRIX(object);
    CUDAFeaturesMatrixPrivate *self_private
        = (CUDAFeaturesMatrixPrivate
               *)(cuda_features_matrix_get_instance_private(self));

    switch(prop_id)
    {
        case PROP_FEATURES_MATRIX_COLS:
            self_private->features_matrix_cols = g_value_get_uint(value);
            break;
        case PROP_FEATURES_MATRIX_ROWS:
            self_private->features_matrix_rows = g_value_get_uint(value);
            break;
        default:
            g_assert_not_reached();
    }
}

/******************************************************************************/
