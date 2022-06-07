/**************************** Includes and Macros *****************************/

#include <gst/cuda/featureextractor/cudafeaturesarray.h>
#include <gst/cuda/featureextractor/cudafeaturescell.h>

#include <glib-object.h>

/************************** Type/Struct Definitions ***************************/

/*************************** Function Declarations ****************************/

/**
 * \brief Callback function for freeing entries within the gpointerarray
 * instance.
 *
 * \details This callback function is used when the gpointerarray instance held
 * by an instance of the CUDAFeaturesArray GObject unref's the gpointerarray.
 * Each individual entry within the gpointerarray is checked to make certain it
 * is a CUDAFeaturesCell GObject instance and is unreferenced.
 *
 * \param[in,out] data The current entry within the gpointerarray instance that
 * has been passed to this callback function.
 */
static void cuda_features_array_cell_free(gpointer data);

/**
 * \brief Constructed callback method for the CUDAFeaturesArray GObject.
 *
 * \details Upon the successful creation of a CUDAFeaturesArray GObject
 * instance, this method is called to provision an array of CUDAFeaturesCell
 * instances. The references to each indivdiual entry of the CUDAFeaturesCell
 * array is held within a pointer array on the CUDAFeaturesArray GObject
 * instance.
 *
 * \param[in,out] object A CUDAFeaturesArray GObject instance to create and
 * store all the CUDAFeaturesCell GObject instance references on.
 */
static void cuda_features_array_constructed(GObject *object);

/**
 * \brief Finaliser method for the CUDAFeaturesArray GObject.
 *
 * \details Upon the destruction of an instance of the CUDAFeaturesArray
 * GObject, this method is called to clean-up the array of references to the
 * CUDAFeaturesCell instances created by this instance of the
 * CUDAFeaturesArray GObject. Unlike the dispose method, the finalise method
 * is only ever called once.
 *
 * \param[in,out] object A CUDAFeaturesArray GObject instance to release all
 * held resources from.
 */
static void cuda_features_array_finalize(GObject *object);

/**
 * \brief Property getter for instances of the CUDAFeaturesArray GObject.
 *
 * \param[in] object A CUDAFeaturesArray GObject instance to get the property from.
 * \param[in] prop_id The ID number for the property to get. This should
 * correspond to an entry in the enum defined in this file.
 * \param[out] value The GValue instance (generic value container) that will
 * contain the value of the property after being converted into the generic
 * GValue type.
 * \param[in] pspec The property specification instance for the property that
 * we are getting. This property specification should exist on the
 * CUDAFeaturesArrayClass GObjectClass structure.
 *
 * \notes Any property ID that does not exist on the CUDAFeaturesArrayClass
 * GClassObject will result in that property being ignored by the getter.
 */
static void cuda_features_array_get_property(
    GObject *object,
    guint prop_id,
    GValue *value,
    GParamSpec *pspec);

/****************************** Static Variables ******************************/

/**
 * \brief Anonymous enumeration containing the list of properties available for
 * the CUDAFeaturesArray GObject type this module defines.
 */
enum
{
    /**
     * \brief ID number for the features-array-length property.
     */
    PROP_FEATURES_ARRAY_LENGTH = 1,

    /**
     * \brief Number of property ID numbers in this enum.
     */
    N_PROPERTIES
};

static const guint32 default_feature_array_length = 40u;

/************************** GObject Type Definitions **************************/

G_DEFINE_TYPE_WITH_PRIVATE(
    CUDAFeaturesArray,
    cuda_features_array,
    G_TYPE_OBJECT);

/**************************** Function Definitions ****************************/

CUDAFeaturesCell *cuda_features_array_at(CUDAFeaturesArray *self, guint32 idx)
{
    CUDAFeaturesCell *cell = NULL;
    CUDAFeaturesArrayPrivate *self_private = NULL;

    self_private
        = (CUDAFeaturesArrayPrivate *)(cuda_features_array_get_instance_private(
            self));

    g_return_val_if_fail(idx < self_private->features_array->len, NULL);

    cell = CUDA_FEATURES_CELL(
        g_ptr_array_index(self_private->features_array, idx));

    return CUDA_FEATURES_CELL(g_object_ref(G_OBJECT(cell)));
}

static void cuda_features_array_cell_free(gpointer data)
{
    if(data != NULL && G_IS_OBJECT(data))
    {
        g_object_unref(data);
    }
}

static void cuda_features_array_class_init(CUDAFeaturesArrayClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    GParamSpec *properties[N_PROPERTIES] = {
        NULL,
    };

    gobject_class->constructed = cuda_features_array_constructed;
    gobject_class->finalize = cuda_features_array_finalize;
    gobject_class->get_property = cuda_features_array_get_property;

    properties[PROP_FEATURES_ARRAY_LENGTH] = g_param_spec_uint(
        "features-array-length",
        "Features Array Length",
        "The number of feature sets within the features array.",
        0,
        G_MAXUINT,
        default_feature_array_length,
        (GParamFlags)(G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

    g_object_class_install_properties(gobject_class, N_PROPERTIES, properties);
}

static void cuda_features_array_constructed(GObject *object)
{
    CUDAFeaturesArray *self = CUDA_FEATURES_ARRAY(object);
    CUDAFeaturesArrayPrivate *self_private
        = (CUDAFeaturesArrayPrivate *)(cuda_features_array_get_instance_private(
            self));

    self_private->features_array
        = g_ptr_array_new_with_free_func(cuda_features_array_cell_free);

    if(G_OBJECT_CLASS(cuda_features_array_parent_class)->constructed != NULL)
    {
        G_OBJECT_CLASS(cuda_features_array_parent_class)->constructed(object);
    }
}

static void cuda_features_array_finalize(GObject *object)
{
    CUDAFeaturesArray *self = CUDA_FEATURES_ARRAY(object);
    CUDAFeaturesArrayPrivate *self_private
        = (CUDAFeaturesArrayPrivate *)(cuda_features_array_get_instance_private(
            self));

    if(self_private->features_array != NULL)
    {
        g_ptr_array_free(self_private->features_array, TRUE);
    }

    if(G_OBJECT_CLASS(cuda_features_array_parent_class)->finalize != NULL)
    {
        G_OBJECT_CLASS(cuda_features_array_parent_class)->finalize(object);
    }
}

static void cuda_features_array_get_property(
    GObject *object,
    guint prop_id,
    GValue *value,
    GParamSpec *pspec)
{
    CUDAFeaturesArray *self = CUDA_FEATURES_ARRAY(object);
    CUDAFeaturesArrayPrivate *self_private
        = (CUDAFeaturesArrayPrivate *)(cuda_features_array_get_instance_private(
            self));

    switch(prop_id)
    {
        case PROP_FEATURES_ARRAY_LENGTH:
            g_value_set_uint(value, self_private->features_array->len);
            break;
        default:
            g_assert_not_reached();
    }
}

static void cuda_features_array_init(CUDAFeaturesArray *self)
{
    CUDAFeaturesArrayPrivate *self_private
        = (CUDAFeaturesArrayPrivate *)(cuda_features_array_get_instance_private(
            self));

    self_private->features_array = NULL;
}

CUDAFeaturesArray *cuda_features_array_new(guint32 length)
{
    CUDAFeaturesArray *features_array
        = CUDA_FEATURES_ARRAY(g_object_new(CUDA_TYPE_FEATURES_ARRAY, NULL));
    CUDAFeaturesArrayPrivate *features_array_private
        = (CUDAFeaturesArrayPrivate *)(cuda_features_array_get_instance_private(
            features_array));

    g_ptr_array_set_size(features_array_private->features_array, length);

    for(guint32 idx = 0; idx < length; idx++)
    {
        CUDAFeaturesCell *cell = g_object_new(CUDA_TYPE_FEATURES_CELL, NULL);
        g_ptr_array_index(features_array_private->features_array, idx) = cell;
    }

    return features_array;
}

/******************************************************************************/
