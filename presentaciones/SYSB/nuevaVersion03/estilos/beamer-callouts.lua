-- Conserva los callouts del QMD como bloques nativos de Beamer.
-- Usa cuadros sin iconos; Quarto puede cargar fontawesome5 igualmente.
function Div(el)
  if not FORMAT:match('beamer') then return nil end
  local kind = nil
  for _, c in ipairs(el.classes) do
    if c:match('^callout%-') then kind = c:gsub('^callout%-', '') end
  end
  if not kind then return nil end
  local names = {note='Nota', tip='Comprobación', important='Concepto clave',
                 warning='Atención', caution='Precaución'}
  local title = names[kind] or 'Concepto'
  local content = el.content
  if #content > 0 and content[1].t == 'Header' then
    title = pandoc.write(pandoc.Pandoc({pandoc.Plain(content[1].content)}), 'latex'):gsub('%s+$','')
    content:remove(1)
  end
  local env = (kind == 'warning' or kind == 'caution') and 'alertblock' or 'block'
  local blocks = {pandoc.RawBlock('latex', '\\begin{'..env..'}{'..title..'}')}
  for _, block in ipairs(content) do blocks[#blocks+1] = block end
  blocks[#blocks+1] = pandoc.RawBlock('latex', '\\end{'..env..'}')
  return blocks
end

function Callout(el)
  if not quarto.doc.is_format('beamer') then return nil end
  local names = {note='Nota', tip='Comprobación', important='Concepto clave',
                 warning='Atención', caution='Precaución'}
  local title = pandoc.utils.stringify(el.title or {})
  if title == '' then title = names[el.type] or 'Concepto' end
  local env = (el.type == 'warning' or el.type == 'caution') and 'alertblock' or 'block'
  local blocks = {pandoc.RawBlock('latex', '\\begin{'..env..'}{'..title..'}')}
  for _, b in ipairs(quarto.utils.as_blocks(el.content)) do blocks[#blocks+1] = b end
  blocks[#blocks+1] = pandoc.RawBlock('latex', '\\end{'..env..'}')
  return pandoc.Div(blocks)
end
