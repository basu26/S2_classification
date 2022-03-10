<?xml version="1.0" encoding="UTF-8"?>
<StyledLayerDescriptor xmlns="http://www.opengis.net/sld" xmlns:ogc="http://www.opengis.net/ogc" xmlns:gml="http://www.opengis.net/gml" xmlns:sld="http://www.opengis.net/sld" version="1.0.0">
  <UserLayer>
    <sld:LayerFeatureConstraints>
      <sld:FeatureTypeConstraint/>
    </sld:LayerFeatureConstraints>
    <sld:UserStyle>
      <sld:Name>WorldLandcover</sld:Name>
      <sld:FeatureTypeStyle>
        <sld:Rule>
          <sld:RasterSymbolizer>
            <sld:ChannelSelection>
              <sld:GrayChannel>
                <sld:SourceChannelName>1</sld:SourceChannelName>
              </sld:GrayChannel>
            </sld:ChannelSelection>
            <sld:ColorMap type="values">
              <sld:ColorMapEntry quantity="1" color="#ffff00" label="Cropland"/>
              <sld:ColorMapEntry quantity="2" color="#966400" label="Shrubland"/>
              <sld:ColorMapEntry quantity="3" color="#00a000" label="Forest"/>
              <sld:ColorMapEntry quantity="4" color="#ffb432" label="Grassland"/>
              <sld:ColorMapEntry quantity="5" color="#c31400" label="Settlement"/>
              <sld:ColorMapEntry quantity="6" color="#ffebaf" label="Sparsely or non-vegetated"/>
              <sld:ColorMapEntry quantity="7" color="#bee8ff" label="Water"/>
              <sld:ColorMapEntry quantity="8" color="#6affff" label="Permanent snow and ice"/>
            </sld:ColorMap>
          </sld:RasterSymbolizer>
        </sld:Rule>
      </sld:FeatureTypeStyle>
    </sld:UserStyle>
  </UserLayer>
</StyledLayerDescriptor>
