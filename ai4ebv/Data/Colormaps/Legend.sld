<?xml version="1.0" encoding="UTF-8"?>
<StyledLayerDescriptor xmlns="http://www.opengis.net/sld" xmlns:gml="http://www.opengis.net/gml" xmlns:sld="http://www.opengis.net/sld" version="1.0.0" xmlns:ogc="http://www.opengis.net/ogc">
  <UserLayer>
    <sld:LayerFeatureConstraints>
      <sld:FeatureTypeConstraint/>
    </sld:LayerFeatureConstraints>
    <sld:UserStyle>
      <sld:Name>RandomForestClassifier_single_FT_2020_M3456789_CORINE_IND_ANN_DEM_N5000</sld:Name>
      <sld:FeatureTypeStyle>
        <sld:Rule>
          <sld:RasterSymbolizer>
            <sld:ChannelSelection>
              <sld:GrayChannel>
                <sld:SourceChannelName>1</sld:SourceChannelName>
              </sld:GrayChannel>
            </sld:ChannelSelection>
            <sld:ColorMap type="values">
              <sld:ColorMapEntry color="#ffff00" quantity="1" label="Crops"/>
              <sld:ColorMapEntry color="#f2a64d" quantity="2" label="Orchards"/>
              <sld:ColorMapEntry color="#e68000" quantity="3" label="Vineyards"/>
              <sld:ColorMapEntry color="#966400" quantity="4" label="Shrubland"/>
              <sld:ColorMapEntry color="#80ff00" quantity="5" label="Broadleaved forest"/>
              <sld:ColorMapEntry color="#00a600" quantity="6" label="Coniferous forest"/>
              <sld:ColorMapEntry color="#ffb432" quantity="7" label="Grassland"/>
              <sld:ColorMapEntry color="#c31400" quantity="8" label="Settlement"/>
              <sld:ColorMapEntry color="#ffebaf" quantity="9" label="Sparsely or non-vegetated"/>
              <sld:ColorMapEntry color="#bee8ff" quantity="10" label="Water"/>
              <sld:ColorMapEntry color="#6affff" quantity="11" label="Permantent snow and ice"/>
            </sld:ColorMap>
          </sld:RasterSymbolizer>
        </sld:Rule>
      </sld:FeatureTypeStyle>
    </sld:UserStyle>
  </UserLayer>
</StyledLayerDescriptor>
