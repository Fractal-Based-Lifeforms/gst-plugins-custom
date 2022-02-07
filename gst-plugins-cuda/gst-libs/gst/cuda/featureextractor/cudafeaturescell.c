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
     * \brief ID number for the Count feature property.
     */
    PROP_COUNT = 1,

    /**
     * \brief ID number for the Pixels feature property.
     */
    PROP_PIXELS,

    /**
     * \brief ID number for the X0ToX1Magnitude feature property.
     */
    PROP_X0_TO_X1_MAGNITUDE,

    /**
     * \brief ID number for the X1ToX0Magnitude feature property.
     */
    PROP_X1_TO_X0_MAGNITUDE,

    /**
     * \brief ID number for the Y0ToY1Magnitude feature property.
     */
    PROP_Y0_TO_Y1_MAGNITUDE,

    /**
     * \brief ID number for the Y1ToY0Magnitude feature property.
     */
    PROP_Y1_TO_Y0_MAGNITUDE,

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

    properties[PROP_COUNT] = g_param_spec_uint(
        "count",
        "Count",
        "The number of optical flow vectors with a squared distance value "
        "greater than the set threshold.",
        0,
        G_MAXUINT,
        0,
        (GParamFlags)(G_PARAM_READWRITE | G_PARAM_CONSTRUCT | G_PARAM_STATIC_STRINGS));

    properties[PROP_PIXELS] = g_param_spec_uint(
        "pixels",
        "Pixels",
        "The total number of pixels contained within the matrix cell.",
        0,
        G_MAXUINT,
        0,
        (GParamFlags)(G_PARAM_READWRITE | G_PARAM_CONSTRUCT | G_PARAM_STATIC_STRINGS));

    properties[PROP_X0_TO_X1_MAGNITUDE] = g_param_spec_float(
        "x0-to-x1-magnitude",
        "X0 to X1 Magnitude",
        "The cumulative absolute value of the positive X-planar values within "
        "the optical flow vectors whose squared magnitude exceed a set "
        "threshold.",
        0.0f,
        G_MAXFLOAT,
        0.0f,
        (GParamFlags)(G_PARAM_READWRITE | G_PARAM_CONSTRUCT | G_PARAM_STATIC_STRINGS));

    properties[PROP_X1_TO_X0_MAGNITUDE] = g_param_spec_float(
        "x1-to-x0-magnitude",
        "X1 to X0 Magnitude",
        "The cumulative absolute value of the negative X-planar values within "
        "the optical flow vectors whose squared magnitude exceed a set "
        "threshold.",
        0.0f,
        G_MAXFLOAT,
        0.0f,
        (GParamFlags)(G_PARAM_READWRITE | G_PARAM_CONSTRUCT | G_PARAM_STATIC_STRINGS));

    properties[PROP_Y0_TO_Y1_MAGNITUDE] = g_param_spec_float(
        "y0-to-y1-magnitude",
        "Y0 to Y1 Magnitude",
        "The cumulative absolute value of the positive Y-planar values within "
        "the optical flow vectors whose squared magnitude exceed a set "
        "threshold.",
        0.0f,
        G_MAXFLOAT,
        0.0f,
        (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

    properties[PROP_Y1_TO_Y0_MAGNITUDE] = g_param_spec_float(
        "y1-to-y0-magnitude",
        "Y1 to Y0 Magnitude",
        "The cumulative absolute value of the negative Y-planar values within "
        "the optical flow vectors whose squared magnitude exceed a set "
        "threshold.",
        0.0f,
        G_MAXFLOAT,
        0.0f,
        (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

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
        case PROP_COUNT:
            g_value_set_uint(value, self_private->count);
            break;
        case PROP_PIXELS:
            g_value_set_uint(value, self_private->pixels);
            break;
        case PROP_X0_TO_X1_MAGNITUDE:
            g_value_set_float(value, self_private->x0_to_x1_magnitude);
            break;
        case PROP_X1_TO_X0_MAGNITUDE:
            g_value_set_float(value, self_private->x1_to_x0_magnitude);
            break;
        case PROP_Y0_TO_Y1_MAGNITUDE:
            g_value_set_float(value, self_private->y0_to_y1_magnitude);
            break;
        case PROP_Y1_TO_Y0_MAGNITUDE:
            g_value_set_float(value, self_private->y1_to_y0_magnitude);
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
        case PROP_COUNT:
            self_private->count = g_value_get_uint(value);
            break;
        case PROP_PIXELS:
            self_private->pixels = g_value_get_uint(value);
            break;
        case PROP_X0_TO_X1_MAGNITUDE:
            self_private->x0_to_x1_magnitude = g_value_get_float(value);
            break;
        case PROP_X1_TO_X0_MAGNITUDE:
            self_private->x1_to_x0_magnitude = g_value_get_float(value);
            break;
        case PROP_Y0_TO_Y1_MAGNITUDE:
            self_private->y0_to_y1_magnitude = g_value_get_float(value);
            break;
        case PROP_Y1_TO_Y0_MAGNITUDE:
            self_private->y1_to_y0_magnitude = g_value_get_float(value);
            break;
        default:
            g_assert_not_reached();
    }
}

/******************************************************************************/
