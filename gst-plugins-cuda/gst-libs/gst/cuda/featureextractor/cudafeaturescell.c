/**************************** Includes and Macros *****************************/

#include <gst/cuda/featureextractor/cudafeaturescell.h>

#include <glib-object.h>

/************************** Type/Struct Definitions ***************************/

/*************************** Function Declarations ****************************/

/**
 * \brief Property getter for instances of the CUDAFeaturesCell GObject.
 *
 * \param[in] gobject A CUDAFeaturesCell GObject instance to get the property
 * from.
 * \param[in] prop_id The ID number for the property to get. This should
 * correspond to an entry in the enum defined in this file.
 * \param[out] value The GValue instance (generic value container) that will
 * contain the value of the property after being converted into the generic
 * GValue type.
 * \param[in] pspec The property specification instance for the property that
 * we are getting. This property specification should exist on the
 *
 * \notes Any property ID that does not exist on the CUDAFeaturesCellClass
 * GClassObject will result in that property being ignored by the getter.
 * CUDAFeaturesCellClass GObjectClass structure.
 */
static void cuda_features_cell_get_property(
    GObject *gobject,
    guint prop_id,
    GValue *value,
    GParamSpec *pspec);

/**
 * \brief Property setter for instances of the CUDAFeaturesCell GObject.
 *
 * \param[in,out] gobject A CUDAFeaturesCell GObject instance to set the
 * property on.
 * \param[in] prop_id The ID number for the property to set. This should
 * correspond to an entry in the enum defined in this file.
 * \param[in] value The GValue instance (generic value container) that will be
 * convered to the correct value type to set the property with.
 * \param[in] pspec The property specification instance for the property that
 * we are setting. This property specification should exist on the
 * CUDAFeaturesCellClass GObjectClass structure.
 *
 * \notes Any property ID that does not exist on the CUDAFeaturesCellClass
 * GClassObject will result in that property being ignored by the setter.
 */
static void cuda_features_cell_set_property(
    GObject *gobject,
    guint prop_id,
    const GValue *value,
    GParamSpec *pspec);

/****************************** Static Variables ******************************/

/**
 * \brief Anonymous enumeration containing the list of properties available for
 * the CUDAFeaturesCell GObject type this module defines.
 */
enum
{
    /**
     * \brief ID number for the SpatialMagnitude feature property.
     */
    PROP_SPATIAL_MAGNITUDE = 1,

    /**
     * \brief Number of property ID numbers in this enum.
     */
    N_PROPERTIES
};

/************************** GObject Type Definitions **************************/

G_DEFINE_TYPE_WITH_PRIVATE(
    CUDAFeaturesCell,
    cuda_features_cell,
    g_object_get_type());

/**************************** Function Definitions ****************************/

static void cuda_features_cell_class_init(CUDAFeaturesCellClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    GParamSpec *properties[N_PROPERTIES] = {
        NULL,
    };

    gobject_class->get_property = cuda_features_cell_get_property;
    gobject_class->set_property = cuda_features_cell_set_property;

    properties[PROP_SPATIAL_MAGNITUDE] = g_param_spec_float(
        "spatial-magnitude",
        "Spatial Magnitude",
        "The cumulative absolute value of the positive/negative X & Y-planar "
        "values within the optical flow vectors whose squared magnitude exceed "
        "a set threshold.",
        0.0f,
        G_MAXFLOAT,
        0.0f,
        (GParamFlags)(G_PARAM_READWRITE | G_PARAM_CONSTRUCT | G_PARAM_STATIC_STRINGS));

    g_object_class_install_properties(gobject_class, N_PROPERTIES, properties);
}

static void cuda_features_cell_get_property(
    GObject *gobject,
    guint prop_id,
    GValue *value,
    GParamSpec *pspec)
{
    CUDAFeaturesCell *self = CUDA_FEATURES_CELL(gobject);
    CUDAFeaturesCellPrivate *self_private
        = (CUDAFeaturesCellPrivate *)(cuda_features_cell_get_instance_private(
            self));

    switch(prop_id)
    {
        case PROP_SPATIAL_MAGNITUDE:
            g_value_set_float(value, self_private->spatial_magnitude);
            break;
        default:
            g_assert_not_reached();
    }
}

static void cuda_features_cell_init(CUDAFeaturesCell *self)
{
}

static void cuda_features_cell_set_property(
    GObject *gobject,
    guint prop_id,
    const GValue *value,
    GParamSpec *pspec)
{
    CUDAFeaturesCell *self = CUDA_FEATURES_CELL(gobject);
    CUDAFeaturesCellPrivate *self_private
        = (CUDAFeaturesCellPrivate *)(cuda_features_cell_get_instance_private(
            self));

    switch(prop_id)
    {
        case PROP_SPATIAL_MAGNITUDE:
            self_private->spatial_magnitude = g_value_get_float(value);
            break;
        default:
            g_assert_not_reached();
    }
}

/******************************************************************************/
